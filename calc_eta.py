from datetime import datetime, timedelta

# 测量依据：
# - batch_size=2，每步比之前 batch=4 快（每步处理数据量减半）
# - 上次 batch=4 时每步 ~0.52s
# - 本次推算：Epoch 19 从约 21:07 开始训练，21:11 到 Step 1800
#   1800 步用了约 4 分钟 = 240 秒 → 每步 0.134s
#   但 batch=2 时步数翻倍（6250步/epoch），所以每epoch实际时间相近

secs_per_step = 240 / 1790      # 0.134 秒/步（实测）

now_epoch     = 19
now_step      = 1800
total_epochs  = 60
steps_per_epoch = 6250

# 剩余步数
remaining_steps = (steps_per_epoch - now_step) + (total_epochs - now_epoch) * steps_per_epoch
remaining_secs  = remaining_steps * secs_per_step
remaining_h     = remaining_secs / 3600
finish_s1       = datetime(2026, 3, 31, 21, 11) + timedelta(seconds=remaining_secs)
finish_s2       = finish_s1 + timedelta(hours=10.5)

print(f"每步耗时:           {secs_per_step:.3f}s  (batch=2, fp32)")
print(f"当前进度:           Epoch {now_epoch}/60, Step {now_step}/6250")
print(f"剩余步数:           {remaining_steps:,} 步")
print(f"剩余时间:           {remaining_h:.1f} 小时")
print(f"Stage 1 完成:       {finish_s1.strftime('%m/%d %H:%M')}")
print(f"Stage 2 (+10.5h):   {finish_s2.strftime('%m/%d %H:%M')}")
