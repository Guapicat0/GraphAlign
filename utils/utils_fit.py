import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
from .utils_training import MSE_loss
from .utils import get_lr
from .utils_predict import compute_score
from .multi_loss import *
from nets.encoder.clip import clip
from .utils_logger import setup_logger
def fit_one_epoch(model_train, model, loss_history, eval_history, optimizer, epoch,
                  epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, loss,
                  fp16, scaler, save_period, save_dir, local_rank=0):
    is_distributed = dist.is_initialized()
    total_loss = 0
    val_loss = 0
    total_predict_scores = []
    total_gt_scores = []

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # Train phase
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        imgs, prompts, gt_scores = batch

        if cuda:
            imgs = imgs.cuda(local_rank)
            gt_scores = gt_scores.cuda(local_rank)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model_train(imgs, prompts)
            loss_value = MSE_loss(outputs, gt_scores)

            if fp16:
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_value.backward()
                optimizer.step()

        # Sync loss across devices
        if is_distributed:
            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            loss_value /= dist.get_world_size()

        total_loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # 初始化分组数据存储
    token_groups = {
        '0-10': {'pred': [], 'gt': [], 'prompts': []},
        '10-20': {'pred': [], 'gt': [], 'prompts': []},
        '20+': {'pred': [], 'gt': [], 'prompts': []}
    }

    # Validation phase
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        imgs, prompts, gt_scores = batch

        if cuda:
            imgs = imgs.cuda(local_rank)
            gt_scores = gt_scores.cuda(local_rank)

        with torch.no_grad():
            outputs = model_train(imgs, prompts)
            loss_value = MSE_loss(outputs, gt_scores)

            # 按token长度分组存储预测结果
            for i, prompt in enumerate(prompts):
                token_count = len(clip.tokenize([prompt])[0].nonzero())
                if token_count <= 10:
                    group = '0-10'
                elif token_count <= 20:
                    group = '10-20'
                else:
                    group = '20+'

                token_groups[group]['pred'].append(outputs[i].detach())
                token_groups[group]['gt'].append(gt_scores[i].detach())
                token_groups[group]['prompts'].append(prompt)

            # Collect predictions and ground truth for overall metrics
            total_predict_scores.append(outputs.detach())
            total_gt_scores.append(gt_scores.detach())

        # Sync validation loss
        if is_distributed:
            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            loss_value /= dist.get_world_size()

        val_loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'val_loss': val_loss / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    # Post-validation processing
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print(f'Epoch:{epoch + 1}/{Epoch}')
        print(f'Total Loss: {total_loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}')

        # 打印分组指标
        print("\nMetrics by Token Length Groups:")
        print("-" * 60)
        print("Group  | Count |  PLCC  |  SROCC  |  KROCC")
        print("-" * 60)

        for group_name, group_data in token_groups.items():
            if len(group_data['pred']) > 0:
                group_pred = torch.stack(group_data['pred']).cpu().numpy().flatten()
                group_gt = torch.stack(group_data['gt']).cpu().numpy().flatten()

                group_results = compute_score(group_pred, group_gt)
                print(
                    f"{group_name:6} | {len(group_data['pred']):5d} | {group_results['plcc']:.4f} | {group_results['srocc']:.4f} | {group_results['krocc']:.4f}")

        print("-" * 60)

        # Save checkpoints
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(
                save_dir,
                f'ep{epoch + 1:03d}-loss{total_loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth'))

        # Save best loss-based model
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Saving best loss model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

    # Metric calculation
    if len(total_predict_scores) > 0 and len(total_gt_scores) > 0:
        # Concatenate predictions and ground truth
        total_predict = torch.cat(total_predict_scores, dim=0).reshape(-1, 1)
        total_gt = torch.cat(total_gt_scores, dim=0).reshape(-1, 1)

        # Gather data from all devices
        if is_distributed:
            world_size = dist.get_world_size()
            # Create containers for gathered data
            gather_predict = [torch.zeros_like(total_predict) for _ in range(world_size)]
            gather_gt = [torch.zeros_like(total_gt) for _ in range(world_size)]
            dist.all_gather(gather_predict, total_predict)
            dist.all_gather(gather_gt, total_gt)

            if local_rank == 0:
                total_predict = torch.cat(gather_predict, dim=0)
                total_gt = torch.cat(gather_gt, dim=0)
        else:
            total_predict = total_predict
            total_gt = total_gt

        # Calculate metrics on main process
        if local_rank == 0:
            # Convert to numpy arrays
            predict_np = total_predict.cpu().numpy().flatten()
            gt_np = total_gt.cpu().numpy().flatten()

            # Compute evaluation metrics
            results = compute_score(predict_np, gt_np)
            print("\nOverall Validation Metrics:")
            print(f"KROCC: {results['krocc']:.4f}, SROCC: {results['srocc']:.4f}, PLCC: {results['plcc']:.4f}")

            # Update evaluation history
            current_avg = (results['krocc'] + results['srocc'] + results['plcc']) / 3
            eval_history.append_eval(epoch + 1, results['krocc'], results['srocc'], results['plcc'])

            # Save best metric-based model
            if current_avg > eval_history.best_avg_metric:
                eval_history.best_avg_metric = current_avg
                print('Saving best metric model to best_eval_weights.pth')
                torch.save(model.state_dict(), os.path.join(save_dir, "best_eval_weights.pth"))

            # Plot metrics
            eval_history.plot_eval()


# def fit_one_epoch(model_train, model, loss_history, eval_history, optimizer, epoch,
#                   epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, loss,
#                   fp16, scaler, save_period, save_dir, local_rank=0):
#     is_distributed = dist.is_initialized()
#     total_loss = 0
#     val_loss = 0
#     total_predict_scores = []
#     total_gt_scores = []
#
#     if local_rank == 0:
#         print('Start Train')
#         pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
#
#     # Train phase
#     model_train.train()
#     for iteration, batch in enumerate(gen):
#         if iteration >= epoch_step:
#             break
#
#         imgs, prompts, gt_scores = batch
#
#         if cuda:
#             imgs = imgs.cuda(local_rank)
#             gt_scores = gt_scores.cuda(local_rank)
#
#         optimizer.zero_grad()
#
#         with torch.set_grad_enabled(True):
#             outputs = model_train(imgs, prompts)
#             loss_value = MSE_loss(outputs, gt_scores)
#
#             if fp16:
#                 scaler.scale(loss_value).backward()
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 loss_value.backward()
#                 optimizer.step()
#
#         # Sync loss across devices
#         if is_distributed:
#             dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
#             loss_value /= dist.get_world_size()
#
#         total_loss += loss_value.item()
#
#         if local_rank == 0:
#             pbar.set_postfix(**{
#                 'total_loss': total_loss / (iteration + 1),
#                 'lr': get_lr(optimizer)
#             })
#             pbar.update(1)
#
#     if local_rank == 0:
#         pbar.close()
#         print('Finish Train')
#         print('Start Validation')
#         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
#
#
#
#
#     # Validation phase
#
#     # 初始化分组数据存储
#     token_groups = {
#         '0-10': {'pred': [], 'gt': [], 'prompts': []},
#         '10-20': {'pred': [], 'gt': [], 'prompts': []},
#         '20+': {'pred': [], 'gt': [], 'prompts': []}
#     }
#
#     model_train.eval()
#     for iteration, batch in enumerate(gen_val):
#         if iteration >= epoch_step_val:
#             break
#
#         imgs, prompts, gt_scores = batch
#
#         if cuda:
#             imgs = imgs.cuda(local_rank)
#             gt_scores = gt_scores.cuda(local_rank)
#
#         with torch.no_grad():
#             outputs = model_train(imgs, prompts)
#             loss_value = MSE_loss(outputs, gt_scores)
#
#             # Collect predictions and ground truth
#             total_predict_scores.append(outputs.detach())
#             total_gt_scores.append(gt_scores.detach())
#
#         # Sync validation loss
#         if is_distributed:
#             dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
#             loss_value /= dist.get_world_size()
#
#         val_loss += loss_value.item()
#
#         if local_rank == 0:
#             pbar.set_postfix(**{
#                 'val_loss': val_loss / (iteration + 1),
#                 'lr': get_lr(optimizer)
#             })
#             pbar.update(1)
#
#     # Post-validation processing
#     if local_rank == 0:
#         pbar.close()
#         print('Finish Validation')
#         loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
#         print(f'Epoch:{epoch + 1}/{Epoch}')
#         print(f'Total Loss: {total_loss / epoch_step:.3f} || Val Loss: {val_loss / epoch_step_val:.3f}')
#
#         # Save checkpoints
#         if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
#             torch.save(model.state_dict(), os.path.join(
#                 save_dir,
#                 f'ep{epoch + 1:03d}-loss{total_loss / epoch_step:.3f}-val_loss{val_loss / epoch_step_val:.3f}.pth'))
#
#         # Save best loss-based model
#         if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
#             print('Saving best loss model to best_epoch_weights.pth')
#             torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
#
#         torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
#
#     # Metric calculation
#     if len(total_predict_scores) > 0 and len(total_gt_scores) > 0:
#         # Concatenate predictions and ground truth
#         total_predict = torch.cat(total_predict_scores, dim=0).reshape(-1, 1)
#         total_gt = torch.cat(total_gt_scores, dim=0).reshape(-1, 1)
#
#         # Gather data from all devices
#         if is_distributed:
#             world_size = dist.get_world_size()
#             # Create containers for gathered data
#             gather_predict = [torch.zeros_like(total_predict) for _ in range(world_size)]
#             gather_gt = [torch.zeros_like(total_gt) for _ in range(world_size)]
#             dist.all_gather(gather_predict, total_predict)
#             dist.all_gather(gather_gt, total_gt)
#
#             if local_rank == 0:
#                 total_predict = torch.cat(gather_predict, dim=0)
#                 total_gt = torch.cat(gather_gt, dim=0)
#         else:
#             total_predict = total_predict
#             total_gt = total_gt
#
#         # Calculate metrics on main process
#         if local_rank == 0:
#             # Convert to numpy arrays
#             predict_np = total_predict.cpu().numpy().flatten()
#             gt_np = total_gt.cpu().numpy().flatten()
#
#             # Compute evaluation metrics
#             results.png = compute_score(predict_np, gt_np)
#             print("Validation Metrics:")
#             print(f"KROCC: {results.png['krocc']:.4f}, SROCC: {results.png['srocc']:.4f}, PLCC: {results.png['plcc']:.4f}")
#
#             # Update evaluation history
#             current_avg = (results.png['krocc'] + results.png['srocc'] + results.png['plcc']) / 3
#             eval_history.append_eval(epoch + 1, results.png['krocc'], results.png['srocc'], results.png['plcc'])
#
#             # Save best metric-based model
#             if current_avg > eval_history.best_avg_metric:
#                 eval_history.best_avg_metric = current_avg
#                 print('Saving best metric model to best_eval_weights.pth')
#                 torch.save(model.state_dict(), os.path.join(save_dir, "best_eval_weights.pth"))
#
#             # Plot metrics
#             eval_history.plot_eval()




# def fit_one_epoch_scene_graph(model_train, model, loss_history, eval_history,optimizer, epoch,
#                   epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, loss,
#                   fp16, scaler, save_period, save_dir, local_rank=0):
#
#     is_distributed = dist.is_initialized()
#     total_loss = 0
#     val_loss = 0
#     total_predict_scores = []
#     total_gt_scores = []
#     total_mse_loss = 0
#     val_total_mse_loss = 0
#
#     # 设置logger
#     if local_rank == 0:
#         logger = setup_logger(save_dir)
#         logger.info(f'Start Train Epoch {epoch + 1}/{Epoch}')
#         pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
#
#     # ---------------------------------------------------------------------#
#     #    Train
#     #
#     # ---------------------------------------------------------------------#
#     model_train.train()
#     for iteration, batch in enumerate(gen):
#         if iteration >= epoch_step:
#             break
#
#         # 根据 batch 的长度动态解包
#         if len(batch) == 6:  # 包含 semantic_graph
#             imgs, prompts, gt_scores, scene_graph, gt_graph, semantic_graph = batch
#         elif len(batch) == 5:  # 不包含 semantic_graph
#             imgs, prompts, gt_scores, scene_graph, gt_graph = batch
#             semantic_graph = None  # 如果没有 semantic_graph，设置为 None
#         else:
#             raise ValueError(f"Unexpected batch structure with length {len(batch)}: {batch}")
#
#         if cuda:
#             imgs = imgs.cuda(local_rank)
#             gt_scores = gt_scores.cuda(local_rank)
#
#         optimizer.zero_grad()
#
#         with torch.set_grad_enabled(True):
#             # 如果 semantic_graph 存在，传递给模型；否则传递 None
#             outputs, gpn_loss = model_train(imgs, prompts, scene_graph, gt_graph, semantic_graph)
#
#
#             # 计算损失
#             mse_loss = MSE_loss(outputs, gt_scores)
#             hml = HarmonicMeanLoss()
#             add = ADDLoss()
#             # 计算总损失
#             loss_value = add(mse_loss, gpn_loss)
# #
#
#             if fp16:
#                 scaler.scale(loss_value).backward()
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 loss_value.backward()
#                 optimizer.step()
#
#         # 修改损失同步部分
#
#
#          # 同步损失值
#         if is_distributed:
#             dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
#             dist.all_reduce(mse_loss, op=dist.ReduceOp.SUM)
#             loss_value /= dist.get_world_size()
#             mse_loss /= dist.get_world_size()
#
#         total_loss += loss_value.item()
#         total_mse_loss += mse_loss.item()
#
#         if local_rank == 0:
#             pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),"mse_loss": total_mse_loss/(iteration + 1),
#                                 'lr': get_lr(optimizer)})
#             pbar.update(1)
#
#     if local_rank == 0:
#         pbar.close()
#         print('Finish Train')
#         print('Start Validation')
#         pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
#
#     # ---------------------------------------------------------------------#
#     #    Validation
#     #
#     # ---------------------------------------------------------------------#
#     drop_last =False
#     if not drop_last :
#         epoch_step_val += 1
#
#     model_train.eval()
#     for iteration, batch in enumerate(gen_val):
#         if iteration >= epoch_step_val:
#             break
#
#         # 根据 batch 的长度动态解包
#         if len(batch) == 6:  # 包含 semantic_graph
#             imgs, prompts, gt_scores, scene_graph, gt_graph, semantic_graph = batch
#         elif len(batch) == 5:  # 不包含 semantic_graph
#             imgs, prompts, gt_scores, scene_graph, gt_graph = batch
#             semantic_graph = None  # 如果没有 semantic_graph，设置为 None
#         else:
#             raise ValueError(f"Unexpected batch structure with length {len(batch)}: {batch}")
#
#         if cuda:
#             imgs = imgs.cuda(local_rank)
#             gt_scores = gt_scores.cuda(local_rank)
#
#         with torch.no_grad():
#             # 如果 semantic_graph 存在，传递给模型；否则传递 None
#             outputs, gpn_loss = model_train(imgs, prompts, scene_graph, gt_graph, semantic_graph)
#
#
#             mse_loss = MSE_loss(outputs, gt_scores)
#             hml = HarmonicMeanLoss()
#             add = ADDLoss()
#             # 计算总损失
#             loss_value = add (loss_value, gpn_loss)
#
#             # 将每个 batch 的输出和真实分数累积起来
#             total_predict_scores.append(outputs)
#             total_gt_scores.append(gt_scores)
#
#
#             # 同步损失值
#             if is_distributed:
#                 dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
#                 dist.all_reduce(mse_loss, op=dist.ReduceOp.SUM)
#                 loss_value /= dist.get_world_size()
#                 mse_loss /= dist.get_world_size()
#
#             val_loss += loss_value.item()
#             val_total_mse_loss += mse_loss.item()
#
#
#         if local_rank == 0:
#             pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),"mse_loss": val_total_mse_loss/(iteration + 1),
#                                 'lr': get_lr(optimizer)})
#             pbar.update(1)
#
#     # ---------------------------------------------------------------------#
#     #    Print loss.Plot loss and Save model
#     #
#     # ---------------------------------------------------------------------#
#     if local_rank == 0:
#         pbar.close()
#         print('Finish Validation')
#         loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
#         print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
#         print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
#
#         if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
#             torch.save(model.state_dict(),
#                        os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
#                            (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))
#
#         if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
#             print('Save best model to best_epoch_weights.pth')
#             torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
#
#         torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
#     # ---------------------------------------------------------------------#
#     #    computer the avg score between the KROCC PLCC,SROcc
#     #
#     # ---------------------------------------------------------------------#
#     # 修改评估指标计算部分
#     if is_distributed:
#         flattened_predict_scores = [t for sublist in total_predict_scores for t in sublist]
#         flattened_gt_scores = [t for sublist in total_gt_scores for t in sublist]
#         result_predict_scores = torch.cat(flattened_predict_scores, dim=0)
#         result_gt_scores = torch.cat(flattened_gt_scores, dim=0)
#
#         local_total_predict_scores = result_predict_scores.reshape(-1, 1)
#         local_total_gt_scores = result_gt_scores.reshape(-1, 1)
#
#         world_size = dist.get_world_size()
#         all_total_predict_scores = [torch.zeros_like(local_total_predict_scores) for _ in range(world_size)]
#         all_total_gt_scores = [torch.zeros_like(local_total_gt_scores) for _ in range(world_size)]
#
#         dist.all_gather(all_total_predict_scores, local_total_predict_scores)
#         dist.all_gather(all_total_gt_scores, local_total_gt_scores)
#         dist.barrier()
#     else:
#         # 非分布式训练时直接处理
#         total_predict_scores = torch.cat(total_predict_scores, dim=0)
#         total_gt_scores = torch.cat(total_gt_scores, dim=0)
#
#     if local_rank == 0:  # 只在一个进程中处理收集的数据
#         # 将所有进程的数据拼接在一起
#         if is_distributed:
#             total_predict_scores = torch.cat(all_total_predict_scores, dim=0)
#             total_gt_scores = torch.cat(all_total_gt_scores, dim=0)
#         # 展平张量
#         total_predict_scores = total_predict_scores.reshape(-1, 1)
#         total_gt_scores = total_gt_scores.reshape(-1, 1)
#
#         # 计算 SROCC, PLCC, KROCC
#         total_predict_score = total_predict_scores.cpu()
#         total_gt_scores = total_gt_scores.cpu()
#         results.png = compute_score(total_predict_score, total_gt_scores)
#         print("Validation Results:")
#         print("KROCC:", results.png["krocc"], "SROCC:", results.png["srocc"], "PLCC:", results.png["plcc"])
#         # 更新评估历史记录
#         eval_history.append_eval(epoch + 1, results.png['krocc'], results.png['srocc'], results.png['plcc'])
#         # 绘制评估指标变化曲线
#         eval_history.plot_eval()
#         # 保存最佳模型
#         if eval_history.best_avg_metric == (results.png['krocc'] + results.png['srocc'] + results.png['plcc']) / 3:
#             print('Save best model to best_eval_weights.pth')
#
#         torch.save(model.state_dict(), os.path.join(save_dir, "best_eval_weights.pth"))
def fit_one_epoch_scene_graph(model_train, model, loss_history, eval_history, optimizer, epoch,
                              epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, loss,
                              fp16, scaler, save_period, save_dir, local_rank=0):
    is_distributed = dist.is_initialized()
    total_loss = 0
    val_loss = 0
    total_predict_scores = []
    total_gt_scores = []
    total_mse_loss = 0
    val_total_mse_loss = 0

    # 设置logger
    if local_rank == 0:
        logger = setup_logger(save_dir)
        logger.info(f'\n{"=" * 50}')
        logger.info(f'Starting new training session - Epoch {epoch + 1}/{Epoch}')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # ---------------------------------------------------------------------#
    #    Train
    # ---------------------------------------------------------------------#
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        # 根据 batch 的长度动态解包
        if len(batch) == 6:  # 包含 semantic_graph
            imgs, prompts, gt_scores, scene_graph, gt_graph, semantic_graph = batch
        elif len(batch) == 5:  # 不包含 semantic_graph
            imgs, prompts, gt_scores, scene_graph, gt_graph = batch
            semantic_graph = None
        else:
            error_msg = f"Unexpected batch structure with length {len(batch)}"
            if local_rank == 0:
                logger.error(error_msg)
            raise ValueError(error_msg)

        if cuda:
            imgs = imgs.cuda(local_rank)
            gt_scores = gt_scores.cuda(local_rank)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs, gpn_loss = model_train(imgs, prompts, scene_graph, gt_graph, semantic_graph)
            mse_loss = MSE_loss(outputs, gt_scores)
            add = ADDLoss()
            loss_value = add(mse_loss, gpn_loss)

            if fp16:
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_value.backward()
                optimizer.step()

        if is_distributed:
            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse_loss, op=dist.ReduceOp.SUM)
            loss_value /= dist.get_world_size()
            mse_loss /= dist.get_world_size()

        total_loss += loss_value.item()
        total_mse_loss += mse_loss.item()

        if local_rank == 0:
            current_loss = total_loss / (iteration + 1)
            current_mse = total_mse_loss / (iteration + 1)
            current_lr = get_lr(optimizer)
            if iteration % 10 == 0:
                logger.info(f'Training - Iteration {iteration + 1}/{epoch_step}, '
                            f'Loss: {current_loss:.4f}, MSE: {current_mse:.4f}, LR: {current_lr:.6f}')
            pbar.set_postfix(**{'total_loss': current_loss,
                                'mse_loss': current_mse,
                                'lr': current_lr})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        logger.info('Finish Train')
        logger.info('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # ---------------------------------------------------------------------#
    #    Validation
    # ---------------------------------------------------------------------#
    drop_last = False
    if not drop_last:
        epoch_step_val += 1

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        if len(batch) == 6:
            imgs, prompts, gt_scores, scene_graph, gt_graph, semantic_graph = batch
        elif len(batch) == 5:
            imgs, prompts, gt_scores, scene_graph, gt_graph = batch
            semantic_graph = None
        else:
            error_msg = f"Unexpected validation batch structure with length {len(batch)}"
            if local_rank == 0:
                logger.error(error_msg)
            raise ValueError(error_msg)

        if cuda:
            imgs = imgs.cuda(local_rank)
            gt_scores = gt_scores.cuda(local_rank)

        with torch.no_grad():
            outputs, gpn_loss = model_train(imgs, prompts, scene_graph, gt_graph, semantic_graph)
            mse_loss = MSE_loss(outputs, gt_scores)
            add = ADDLoss()
            loss_value = add(mse_loss, gpn_loss)

            total_predict_scores.append(outputs)
            total_gt_scores.append(gt_scores)

            if is_distributed:
                dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                dist.all_reduce(mse_loss, op=dist.ReduceOp.SUM)
                loss_value /= dist.get_world_size()
                mse_loss /= dist.get_world_size()

            val_loss += loss_value.item()
            val_total_mse_loss += mse_loss.item()

        if local_rank == 0:
            current_val_loss = val_loss / (iteration + 1)
            current_val_mse = val_total_mse_loss / (iteration + 1)
            if iteration % 10 == 0:
                logger.info(f'Validation - Iteration {iteration + 1}/{epoch_step_val}, '
                            f'Loss: {current_val_loss:.4f}, MSE: {current_val_mse:.4f}')
            pbar.set_postfix(**{'val_loss': current_val_loss,
                                'mse_loss': current_val_mse,
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # ---------------------------------------------------------------------#
    #    Compute Metrics and Save Results
    # ---------------------------------------------------------------------#
    if is_distributed:
        flattened_predict_scores = [t for sublist in total_predict_scores for t in sublist]
        flattened_gt_scores = [t for sublist in total_gt_scores for t in sublist]
        result_predict_scores = torch.cat(flattened_predict_scores, dim=0)
        result_gt_scores = torch.cat(flattened_gt_scores, dim=0)

        local_total_predict_scores = result_predict_scores.reshape(-1, 1)
        local_total_gt_scores = result_gt_scores.reshape(-1, 1)

        world_size = dist.get_world_size()
        all_total_predict_scores = [torch.zeros_like(local_total_predict_scores) for _ in range(world_size)]
        all_total_gt_scores = [torch.zeros_like(local_total_gt_scores) for _ in range(world_size)]

        dist.all_gather(all_total_predict_scores, local_total_predict_scores)
        dist.all_gather(all_total_gt_scores, local_total_gt_scores)
        dist.barrier()
    else:
        total_predict_scores = torch.cat(total_predict_scores, dim=0)
        total_gt_scores = torch.cat(total_gt_scores, dim=0)

    if local_rank == 0:
        pbar.close()

        if is_distributed:
            total_predict_scores = torch.cat(all_total_predict_scores, dim=0)
            total_gt_scores = torch.cat(all_total_gt_scores, dim=0)

        total_predict_scores = total_predict_scores.reshape(-1, 1)
        total_gt_scores = total_gt_scores.reshape(-1, 1)

        total_predict_score = total_predict_scores.cpu()
        total_gt_scores = total_gt_scores.cpu()
        results = compute_score(total_predict_score, total_gt_scores)

        # 计算平均损失
        avg_train_loss = total_loss / epoch_step
        avg_val_loss = val_loss / epoch_step_val
        avg_metric = (results['krocc'] + results['srocc'] + results['plcc']) / 3

        # 记录epoch总结
        logger.info(f'\nEpoch {epoch + 1}/{Epoch} Summary:')
        logger.info(f'Average Train Loss: {avg_train_loss:.4f}')
        logger.info(f'Average Val Loss: {avg_val_loss:.4f}')
        logger.info(f'Metrics - KROCC: {results["krocc"]:.4f}, '
                    f'SROCC: {results["srocc"]:.4f}, '
                    f'PLCC: {results["plcc"]:.4f}')
        logger.info(f'Average Metric: {avg_metric:.4f}')

        # 更新历史记录
        loss_history.append_loss(epoch + 1, avg_train_loss, avg_val_loss)
        eval_history.append_eval(epoch + 1, results['krocc'], results['srocc'], results['plcc'])
        eval_history.plot_eval()

        # 保存模型
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            model_path = os.path.join(save_dir,
                                      f'ep{epoch + 1:03d}-loss{avg_train_loss:.3f}-val_loss{avg_val_loss:.3f}.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f'Saved periodic model to {model_path}')

        if len(loss_history.val_loss) <= 1 or avg_val_loss <= min(loss_history.val_loss):
            best_model_path = os.path.join(save_dir, "best_epoch_weights.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'Saved best loss model to {best_model_path}')

        if eval_history.best_avg_metric == avg_metric:
            best_eval_path = os.path.join(save_dir, "best_eval_weights.pth")
            torch.save(model.state_dict(), best_eval_path)
            logger.info(f'Saved best metrics model to {best_eval_path}')

        # 保存最新模型
        last_model_path = os.path.join(save_dir, "last_epoch_weights.pth")
        torch.save(model.state_dict(), last_model_path)

        logger.info(f'{"=" * 50}\n')  # 添加结束分隔线


def fit_one_epoch_scene_graph_ACC(model_train, model, loss_history, eval_history,optimizer, epoch,
                  epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, loss,
                  fp16, scaler, save_period, save_dir, local_rank=0):

    is_distributed = dist.is_initialized()
    total_loss = 0
    val_loss = 0
    total_predict_scores = []
    total_gt_scores = []
    total_mse_loss = 0
    val_total_mse_loss = 0

    total_gpn_acc = 0  # 新增：GPN准确率
    total_gpn_tp_rate = 0  # 新增：真阳性率
    total_gpn_tn_rate = 0  # 新增：真阴性率
    val_total_gpn_acc = 0
    val_total_gpn_tp_rate = 0
    val_total_gpn_tn_rate = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # ---------------------------------------------------------------------#
    #    Train
    #
    # ---------------------------------------------------------------------#
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        # 根据 batch 的长度动态解包
        if len(batch) == 6:  # 包含 semantic_graph
            imgs, prompts, gt_scores, scene_graph, gt_graph, semantic_graph = batch
        elif len(batch) == 5:  # 不包含 semantic_graph
            imgs, prompts, gt_scores, scene_graph, gt_graph = batch
            semantic_graph = None  # 如果没有 semantic_graph，设置为 None
        else:
            raise ValueError(f"Unexpected batch structure with length {len(batch)}: {batch}")

        if cuda:
            imgs = imgs.cuda(local_rank)
            gt_scores = gt_scores.cuda(local_rank)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # 如果 semantic_graph 存在，传递给模型；否则传递 None
            outputs, gpn_loss,gpn_acc, tp_rate, tn_rate  = model_train(imgs, prompts, scene_graph, gt_graph, semantic_graph)

            # 累积GPN指标
            total_gpn_acc += gpn_acc
            total_gpn_tp_rate += tp_rate
            total_gpn_tn_rate += tn_rate

            # 计算损失
            mse_loss = MSE_loss(outputs, gt_scores)
            hml = HarmonicMeanLoss()
            add = ADDLoss()
            # 计算总损失
            loss_value = add(mse_loss, gpn_loss)
#

            if fp16:
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_value.backward()
                optimizer.step()

        # 修改损失同步部分


         # 同步损失值
        if is_distributed:
            dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
            dist.all_reduce(mse_loss, op=dist.ReduceOp.SUM)
            loss_value /= dist.get_world_size()
            mse_loss /= dist.get_world_size()

        total_loss += loss_value.item()
        total_mse_loss += mse_loss.item()

        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (iteration + 1),
                'mse_loss': total_mse_loss/(iteration + 1),
                'gpn_acc': total_gpn_acc / (iteration + 1),
                'gpn_tp': total_gpn_tp_rate / (iteration + 1),
                'gpn_tn': total_gpn_tn_rate / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # ---------------------------------------------------------------------#
    #    Validation
    #
    # ---------------------------------------------------------------------#
    drop_last =False
    if not drop_last :
        epoch_step_val += 1

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break

        # 根据 batch 的长度动态解包
        if len(batch) == 6:  # 包含 semantic_graph
            imgs, prompts, gt_scores, scene_graph, gt_graph, semantic_graph = batch
        elif len(batch) == 5:  # 不包含 semantic_graph
            imgs, prompts, gt_scores, scene_graph, gt_graph = batch
            semantic_graph = None  # 如果没有 semantic_graph，设置为 None
        else:
            raise ValueError(f"Unexpected batch structure with length {len(batch)}: {batch}")

        if cuda:
            imgs = imgs.cuda(local_rank)
            gt_scores = gt_scores.cuda(local_rank)

        with torch.no_grad():
            # 如果 semantic_graph 存在，传递给模型；否则传递 None
            outputs, gpn_loss, gpn_acc, tp_rate, tn_rate = model_train(imgs, prompts, scene_graph, gt_graph,
                                                                       semantic_graph)

            # 累积验证集GPN指标
            val_total_gpn_acc += gpn_acc
            val_total_gpn_tp_rate += tp_rate
            val_total_gpn_tn_rate += tn_rate


            mse_loss = MSE_loss(outputs, gt_scores)
            hml = HarmonicMeanLoss()
            add = ADDLoss()
            # 计算总损失
            loss_value = add (loss_value, gpn_loss)

            # 将每个 batch 的输出和真实分数累积起来
            total_predict_scores.append(outputs)
            total_gt_scores.append(gt_scores)


            # 同步损失值
            if is_distributed:
                dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                dist.all_reduce(mse_loss, op=dist.ReduceOp.SUM)
                loss_value /= dist.get_world_size()
                mse_loss /= dist.get_world_size()

            val_loss += loss_value.item()
            val_total_mse_loss += mse_loss.item()



        if local_rank == 0:
            pbar.set_postfix(**{
                'val_loss': val_loss / (iteration + 1),
                'val_mse': val_total_mse_loss/(iteration + 1),
                'val_gpn_acc': val_total_gpn_acc / (iteration + 1),
                'val_gpn_tp': val_total_gpn_tp_rate / (iteration + 1),
                'val_gpn_tn': val_total_gpn_tn_rate / (iteration + 1),
                'lr': get_lr(optimizer)
            })
            pbar.update(1)
    # ---------------------------------------------------------------------#
    #    Print loss.Plot loss and Save model
    #
    # ---------------------------------------------------------------------#
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        print('GPN Metrics:')
        print('Train - ACC: %.3f, TP Rate: %.3f, TN Rate: %.3f' % (
            total_gpn_acc / epoch_step,
            total_gpn_tp_rate / epoch_step,
            total_gpn_tn_rate / epoch_step
        ))
        print('Val - ACC: %.3f, TP Rate: %.3f, TN Rate: %.3f' % (
            val_total_gpn_acc / epoch_step_val,
            val_total_gpn_tp_rate / epoch_step_val,
            val_total_gpn_tn_rate / epoch_step_val
        ))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
                           (epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
    # ---------------------------------------------------------------------#
    #    computer the avg score between the KROCC PLCC,SROcc
    #
    # ---------------------------------------------------------------------#
    # 修改评估指标计算部分
    if is_distributed:
        flattened_predict_scores = [t for sublist in total_predict_scores for t in sublist]
        flattened_gt_scores = [t for sublist in total_gt_scores for t in sublist]
        result_predict_scores = torch.cat(flattened_predict_scores, dim=0)
        result_gt_scores = torch.cat(flattened_gt_scores, dim=0)

        local_total_predict_scores = result_predict_scores.reshape(-1, 1)
        local_total_gt_scores = result_gt_scores.reshape(-1, 1)

        world_size = dist.get_world_size()
        all_total_predict_scores = [torch.zeros_like(local_total_predict_scores) for _ in range(world_size)]
        all_total_gt_scores = [torch.zeros_like(local_total_gt_scores) for _ in range(world_size)]

        dist.all_gather(all_total_predict_scores, local_total_predict_scores)
        dist.all_gather(all_total_gt_scores, local_total_gt_scores)
        dist.barrier()
    else:
        # 非分布式训练时直接处理
        total_predict_scores = torch.cat(total_predict_scores, dim=0)
        total_gt_scores = torch.cat(total_gt_scores, dim=0)

    if local_rank == 0:  # 只在一个进程中处理收集的数据
        # 将所有进程的数据拼接在一起
        if is_distributed:
            total_predict_scores = torch.cat(all_total_predict_scores, dim=0)
            total_gt_scores = torch.cat(all_total_gt_scores, dim=0)
        # 展平张量
        total_predict_scores = total_predict_scores.reshape(-1, 1)
        total_gt_scores = total_gt_scores.reshape(-1, 1)

        # 计算 SROCC, PLCC, KROCC
        total_predict_score = total_predict_scores.cpu()
        total_gt_scores = total_gt_scores.cpu()
        results = compute_score(total_predict_score, total_gt_scores)
        print("Validation Results:")
        print("KROCC:", results["krocc"], "SROCC:", results["srocc"], "PLCC:", results["plcc"])
        # 更新评估历史记录
        eval_history.append_eval(epoch + 1, results['krocc'], results['srocc'], results['plcc'])
        # 绘制评估指标变化曲线
        eval_history.plot_eval()
        # 保存最佳模型
        if eval_history.best_avg_metric == (results['krocc'] + results['srocc'] + results['plcc']) / 3:
            print('Save best model to best_eval_weights.pth')

        torch.save(model.state_dict(), os.path.join(save_dir, "best_eval_weights.pth"))





