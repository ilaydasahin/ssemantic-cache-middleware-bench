# Quick Start: 20 Free Gemini Keys

**Goal**: Run full benchmark suite (10K+ queries) completely free using 20 Gemini API keys.

## 1. Get 20 API Keys (5 minutes)

Visit: https://aistudio.google.com/app/apikey

- Create 20 keys (you can use different Google accounts)
- Each key is free: 15 RPM, 1,500 RPD
- Total capacity: ~240 RPM, ~29,000 RPD

## 2. Configure Keys (1 minute)

```bash
# Copy example file
cp .env.example .env

# Edit .env and paste your 20 keys
nano .env

# Load environment
source .env
```

Or directly:

```bash
export GEMINI_API_KEYS="key1,key2,key3,...,key20"
```

## 3. Test Setup (2 minutes)

```bash
# Quick test with 50 queries
bash test_multi_key.sh
```

Expected output:
```
✅ Found 20 API keys
📊 Estimated capacity:
   - Rate: ~240 requests/minute
   - Daily: ~29,000 requests/day
🧪 Running test with 50 queries...
✅ Test completed!
```

## 4. Run Full Benchmark (40-60 minutes)

```bash
# Full suite: ~10K queries across 3 datasets
bash run_full_benchmark_suite.sh
```

Progress logs:
```
INFO: ✅ Multi-key mode: 20 keys detected. Total capacity: ~240RPM, ~29000RPD
INFO: Progress: 100 total calls across 20 keys (avg 5/key)
INFO: Progress: 200 total calls across 20 keys (avg 10/key)
...
INFO: Progress: 10000 total calls across 20 keys (avg 500/key)
```

## What Happens Automatically

✅ **Rate Limiting**: Each key waits 4.8s between calls  
✅ **Quota Tracking**: Stops at 1,450 calls per key (safe buffer)  
✅ **Key Rotation**: Automatically switches when a key is exhausted  
✅ **Parallel Execution**: Up to 20 concurrent requests  
✅ **Progress Logging**: Updates every 100 calls  

## Quota Management

| Dataset Size | Keys Used | Time Required |
|--------------|-----------|---------------|
| 1,000 queries | 1 key | ~70 min |
| 5,000 queries | 4 keys | ~90 min |
| 10,000 queries | 7 keys | ~180 min |
| 20,000 queries | 14 keys | ~360 min |
| 29,000 queries | 20 keys | ~500 min |

## If Quota Exhausted

```
ERROR: ALL 20 keys exhausted daily quota. Total: 29000 calls today.
```

**Solutions**:
1. Wait until tomorrow (quotas reset at midnight PST)
2. Use mock mode: `-Dspring-boot.run.profiles=benchmark,benchmark-mock`
3. Reduce sample size: `-Dbenchmark.sample-size=5000`

## Cost: $0.00

With 20 free keys, you can run the entire benchmark suite without any cost!

## Need Help?

- Full guide: [MULTI_KEY_SETUP.md](MULTI_KEY_SETUP.md)
- Troubleshooting: See "Troubleshooting" section in MULTI_KEY_SETUP.md
- Issues: Open a GitHub issue

---

**Ready?** Just run `bash test_multi_key.sh` to get started!
