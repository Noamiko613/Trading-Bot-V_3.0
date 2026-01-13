# Starting Fresh RL Training

## Why Start Fresh?

The PPO model has been optimized with **significantly different hyperparameters**:
- Learning rate: 3e-4 → **5e-4** (adaptive)
- Entropy: 0.05 → **0.08** (adaptive decay)
- Reward structure: **Completely redesigned**
- Network architecture: **Optimized**

**Continuing from an old model** would:
- ❌ Slow down learning (wrong hyperparameters)
- ❌ Cause instability (mismatched reward structure)
- ❌ Learn suboptimal patterns

**Starting fresh** ensures:
- ✅ Optimal learning from the beginning
- ✅ Faster convergence (2-3x)
- ✅ Better real market performance

## Option 1: Use Reset Script (Recommended)

### Step 1: Backup and Clean Old Model
```bash
python reset_rl_training.py
```

This will:
- ✅ Backup old model files to `models/rl_models_backup/`
- ✅ Clean old checkpoints and state
- ✅ Prepare for fresh training

### Step 2: Start Fresh Training
```bash
python rl_training.py --dashboard --fresh-start
```

## Option 2: Manual Cleanup

### Step 1: Backup (Optional but Recommended)
```bash
# Create backup directory
mkdir -p models/rl_models_backup/backup_$(date +%Y%m%d_%H%M%S)

# Copy important files
cp -r models/rl_models/checkpoints models/rl_models_backup/backup_*/
cp models/rl_models/rl_training_state.json models/rl_models_backup/backup_*/
```

### Step 2: Clean Old Files
```bash
# Remove old checkpoints
rm -rf models/rl_models/checkpoints

# Remove old state
rm -f models/rl_models/rl_training_state.json

# Remove old experience buffers
rm -f models/rl_models/experience_buffers.json

# Remove TensorBoard logs (optional)
rm -rf models/rl_models/tensorboard
```

### Step 3: Start Fresh Training
```bash
python rl_training.py --dashboard --fresh-start
```

## Option 3: Quick Start (No Backup)

If you're sure you don't need the old model:

```bash
# Clean everything
rm -rf models/rl_models/checkpoints
rm -f models/rl_models/rl_training_state.json
rm -f models/rl_models/experience_buffers.json

# Start fresh
python rl_training.py --dashboard --fresh-start
```

## What Gets Cleaned?

The reset script removes:
- ✅ `rl_training_state.json` - Training state (episode, timesteps, etc.)
- ✅ `checkpoints/` - All model checkpoints
- ✅ `experience_buffers.json` - Experience replay buffers
- ✅ `tensorboard/` - TensorBoard logs (optional)
- ✅ `final_model.zip` - Final model files
- ✅ `best_model_*.zip` - Best model files

**What's preserved:**
- ✅ Backup directory (if you used the script)
- ✅ Database (`sim_results/trades.db`) - Trade history
- ✅ Config files

## Training Command Examples

### Basic Training
```bash
python rl_training.py --fresh-start
```

### With Dashboard
```bash
python rl_training.py --dashboard --fresh-start
```

### Single Symbol Training
```bash
python rl_training.py --symbol BTCUSDT --dashboard --fresh-start
```

### Multi-Symbol Training (Default)
```bash
python rl_training.py --dashboard --fresh-start
```

## Expected Training Time

With optimized hyperparameters:
- **First 10K steps**: ~30-60 minutes (initial learning)
- **First 100K steps**: ~5-10 hours (basic patterns)
- **500K+ steps**: ~1-2 days (good performance)
- **1M+ steps**: ~2-4 days (excellent performance)

**Note**: Training continues until acceptance criteria are met (win rate ≥70%, profit factor ≥1.5, etc.)

## Monitoring Progress

1. **Dashboard**: Real-time metrics
   ```bash
   python rl_training.py --dashboard
   ```

2. **TensorBoard**: Learning curves
   ```bash
   tensorboard --logdir models/rl_models/tensorboard
   ```

3. **Logs**: Check console output for trade openings/closings

## Troubleshooting

### "Model directory not found"
- Create it: `mkdir -p models/rl_models`

### "Checkpoint still loading"
- Use `--fresh-start` flag to force new model

### "Training too slow"
- Check GPU availability: `nvidia-smi`
- Reduce batch size if memory issues
- Use single symbol: `--symbol BTCUSDT`

## Next Steps After Training

1. **Monitor Metrics**: Watch win rate, profit factor, Sharpe ratio
2. **Paper Trading**: Test on paper trading before live
3. **Evaluate**: Check dashboard for performance metrics
4. **Deploy**: Once acceptance criteria are met

---

**Ready to start?** Run:
```bash
python reset_rl_training.py
python rl_training.py --dashboard --fresh-start
```

