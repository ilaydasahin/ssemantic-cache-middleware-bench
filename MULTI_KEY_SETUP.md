# Multi-Key Setup Guide - 20 Gemini Keys (Free Tier)

## Overview

This guide explains how to run experiments using 20 free Gemini API keys to maximize throughput while staying within quota limits.

## Free Tier Limits (Per Key)

- **Rate Limit**: 15 RPM (requests per minute)
- **Daily Quota**: 1,500 RPD (requests per day)
- **Total with 20 keys**: ~240 RPM, ~29,000 RPD (safe buffer)

## Setup

### 1. Get Your API Keys

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create 20 API keys (you can use different Google accounts if needed)
3. Copy all keys

### 2. Configure Keys

Create a file with your keys (one per line or comma-separated):

```bash
# Option 1: Environment variable (recommended)
export GEMINI_API_KEYS="key1,key2,key3,...,key20"

# Option 2: In application.yml
llm:
  api-keys: "key1,key2,key3,...,key20"
```

**Security Note**: Never commit keys to git! Use environment variables or a separate config file.

### 3. Verify Configuration

```bash
mvn spring-boot:run -Dspring-boot.run.profiles=benchmark,benchmark-mock

# Look for this in logs:
# ✅ Multi-key mode: 20 keys detected. Total capacity: ~240RPM, ~29000RPD (free tier safe)
```

## How It Works

### Intelligent Key Rotation

The system automatically:

1. **Rate Limiting**: Each key waits 4.8s between calls (12.5 RPM, safe buffer)
2. **Daily Quota Tracking**: Stops using a key after 1,450 calls (safe buffer for 1,500 limit)
3. **Parallel Execution**: Up to 20 concurrent requests (one per key)
4. **Automatic Failover**: If a key hits quota, rotates to next available key

### Example Timeline

```
Time    Key1   Key2   Key3   ...  Key20
0.0s    Call1  Call2  Call3  ...  Call20
4.8s    Call21 Call22 Call23 ...  Call40
9.6s    Call41 Call42 Call43 ...  Call60
...
```

**Result**: ~240 calls/minute instead of 15 calls/minute with single key!

## Running Experiments

### Full Benchmark Suite (10K queries)

```bash
# With 20 keys: ~42 minutes (240 RPM)
# With 1 key: ~11 hours (15 RPM)

bash run_full_benchmark_suite.sh
```

### Single Dataset Test

```bash
mvn spring-boot:run \
  -Dspring-boot.run.profiles=benchmark \
  -Dbenchmark.current-dataset=msmarco \
  -Dbenchmark.current-seed=42 \
  -Dbenchmark.sample-size=1000 \
  -Dbenchmark.output-file=results/test.json
```

### Monitor Progress

The system logs progress every 100 calls:

```
INFO: Progress: 100 total calls across 20 keys (avg 5/key)
INFO: Progress: 200 total calls across 20 keys (avg 10/key)
...
INFO: Progress: 10000 total calls across 20 keys (avg 500/key)
```

## Quota Management

### Daily Limits

With 20 keys, you can make **~29,000 calls per day** safely:

```
Dataset Size    Keys Needed    Time Required
1,000 queries   1 key          ~70 minutes
5,000 queries   4 keys         ~90 minutes
10,000 queries  7 keys         ~180 minutes
20,000 queries  14 keys        ~360 minutes
29,000 queries  20 keys        ~500 minutes
```

### What Happens When Quota Exhausted?

```
WARN: Key 5 hit quota/rate limit. Usage: 1450/1450. Rotating...
INFO: Progress: 15000 total calls across 20 keys (avg 750/key)
...
ERROR: ALL 20 keys exhausted daily quota (1450 calls each). Total: 29000 calls today.
```

The system will:
1. Stop gracefully
2. Save partial results
3. Log total usage per key

### Resume Next Day

Daily quotas reset at midnight PST. Simply restart the experiment:

```bash
# Results are saved incrementally, so you can continue
bash run_full_benchmark_suite.sh
```

## Cost Estimation

### Free Tier (Current Setup)

- **Cost**: $0.00
- **Capacity**: 29,000 calls/day
- **Typical query**: ~100 tokens input, ~200 tokens output
- **Total tokens/day**: ~8.7M tokens (well within free tier)

### If You Need More

Gemini Flash pricing (if you exceed free tier):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

For 100K queries:
- Estimated cost: ~$2.60
- Still very cheap compared to GPT-4!

## Troubleshooting

### "No API keys provided"

```bash
# Check environment variable
echo $GEMINI_API_KEYS

# Or check application.yml
grep "api-keys" src/main/resources/application.yml
```

### "Key X hit quota limit"

This is normal! The system will rotate to the next key automatically.

### "ALL keys exhausted"

You've used all 29,000 calls for today. Options:
1. Wait until tomorrow (quotas reset at midnight PST)
2. Use `benchmark-mock` profile for testing (no API calls)
3. Reduce sample size: `-Dbenchmark.sample-size=5000`

### Rate Limit Errors (429)

The system handles this automatically by:
1. Marking the key as exhausted
2. Rotating to next available key
3. Retrying the request

## Best Practices

### 1. Start Small

```bash
# Test with 100 queries first
-Dbenchmark.sample-size=100
```

### 2. Use Mock Mode for Development

```bash
# No API calls, instant results
-Dspring-boot.run.profiles=benchmark,benchmark-mock
```

### 3. Monitor Logs

```bash
# Watch for quota warnings
tail -f logs/benchmark.log | grep -i "quota\|exhausted\|progress"
```

### 4. Batch Experiments

Run multiple experiments in sequence:

```bash
for seed in 42 123 456 789 1024; do
  mvn spring-boot:run \
    -Dspring-boot.run.profiles=benchmark \
    -Dbenchmark.current-seed=$seed \
    -Dbenchmark.output-file=results/run_$seed.json
done
```

## Security Checklist

- [ ] Keys stored in environment variables (not in code)
- [ ] `.gitignore` includes `application-local.yml`
- [ ] Never commit keys to version control
- [ ] Rotate keys periodically
- [ ] Monitor usage in [Google AI Studio](https://aistudio.google.com/app/apikey)

## FAQ

**Q: Can I use more than 20 keys?**  
A: Yes! The system supports unlimited keys. Just add them to `GEMINI_API_KEYS`.

**Q: What if I only have 5 keys?**  
A: Still works! You'll get ~60 RPM instead of 240 RPM. Adjust expectations accordingly.

**Q: Do I need to restart if a key fails?**  
A: No! The system automatically rotates to the next available key.

**Q: How do I check my quota usage?**  
A: Check logs for "Progress" messages or visit [Google AI Studio](https://aistudio.google.com/app/apikey).

**Q: Can I mix free and paid keys?**  
A: Yes! Paid keys have higher limits (1000 RPM, 4M RPD) and will be used more frequently.

## Summary

With 20 free Gemini keys, you can:
- ✅ Run full benchmark suite (~10K queries) in ~42 minutes
- ✅ Make ~29,000 API calls per day
- ✅ Stay completely within free tier limits
- ✅ Automatic failover and quota management
- ✅ Zero cost for academic research

**Ready to start?** Just set `GEMINI_API_KEYS` and run your experiments!
