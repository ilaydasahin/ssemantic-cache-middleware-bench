// K6 Load Test Script for Semantic Cache
// Usage: k6 run scripts/load-test.js

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cacheHitRate = new Rate('cache_hits');
const latencyTrend = new Trend('latency_ms');

export const options = {
  stages: [
    { duration: '1m', target: 50 },   // Warm up
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 500 },  // Ramp up to 500 users
    { duration: '5m', target: 500 },  // Stay at 500 users
    { duration: '2m', target: 1000 }, // Ramp up to 1000 users
    { duration: '5m', target: 1000 }, // Stay at 1000 users
    { duration: '2m', target: 0 },    // Ramp down
  ],
  thresholds: {
    'http_req_duration': ['p(95)<500', 'p(99)<1000'], // 95% < 500ms, 99% < 1s
    'http_req_failed': ['rate<0.01'],                  // Error rate < 1%
    'errors': ['rate<0.01'],
    'cache_hits': ['rate>0.80'],                       // Hit rate > 80%
  },
};

// Sample queries (mix of similar and different queries for cache testing)
const queries = [
  // Group 1: Machine Learning (similar queries)
  "What is machine learning?",
  "Explain machine learning",
  "How does machine learning work?",
  "Define machine learning",
  
  // Group 2: Neural Networks (similar queries)
  "What are neural networks?",
  "Explain neural networks",
  "How do neural networks work?",
  "Define neural networks",
  
  // Group 3: Semantic Caching (similar queries)
  "What is semantic caching?",
  "Explain semantic caching",
  "How does semantic caching work?",
  "Define semantic caching",
  
  // Group 4: Database (similar queries)
  "How to optimize database queries?",
  "Database query optimization techniques",
  "Best practices for database optimization",
  "Optimize SQL queries",
  
  // Group 5: AI vs ML (similar queries)
  "What is the difference between AI and ML?",
  "AI vs ML comparison",
  "Difference between artificial intelligence and machine learning",
  "Compare AI and ML",
];

export default function () {
  // Select a random query (biased towards repeated queries for cache testing)
  const queryIndex = Math.random() < 0.7 
    ? Math.floor(Math.random() * 5) // 70% chance of top 5 queries (high cache hit)
    : Math.floor(Math.random() * queries.length); // 30% chance of any query
  
  const query = queries[queryIndex];
  
  const payload = JSON.stringify({
    query: query,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const startTime = new Date().getTime();
  const res = http.post('http://localhost:8080/api/query', payload, params);
  const endTime = new Date().getTime();
  const latency = endTime - startTime;

  // Record metrics
  latencyTrend.add(latency);
  errorRate.add(res.status !== 200);

  // Check if response indicates cache hit
  if (res.status === 200 && res.body) {
    try {
      const body = JSON.parse(res.body);
      if (body.cacheHit !== undefined) {
        cacheHitRate.add(body.cacheHit);
      }
    } catch (e) {
      // Ignore JSON parse errors
    }
  }

  // Assertions
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => latency < 500,
    'response time < 1000ms': (r) => latency < 1000,
    'has response body': (r) => r.body && r.body.length > 0,
  });

  // Think time (simulate real user behavior)
  sleep(Math.random() * 2 + 1); // 1-3 seconds
}

export function handleSummary(data) {
  return {
    'load-test-summary.json': JSON.stringify(data, null, 2),
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options) {
  const indent = options.indent || '';
  const enableColors = options.enableColors || false;
  
  let summary = '\n';
  summary += indent + '='.repeat(80) + '\n';
  summary += indent + '                    LOAD TEST SUMMARY\n';
  summary += indent + '='.repeat(80) + '\n\n';
  
  // HTTP metrics
  summary += indent + 'HTTP Metrics:\n';
  summary += indent + `  Total Requests: ${data.metrics.http_reqs.values.count}\n`;
  summary += indent + `  Failed Requests: ${data.metrics.http_req_failed.values.rate * 100}%\n`;
  summary += indent + `  Requests/sec: ${data.metrics.http_reqs.values.rate.toFixed(2)}\n\n`;
  
  // Latency metrics
  summary += indent + 'Latency:\n';
  summary += indent + `  P50: ${data.metrics.http_req_duration.values['p(50)']}ms\n`;
  summary += indent + `  P95: ${data.metrics.http_req_duration.values['p(95)']}ms\n`;
  summary += indent + `  P99: ${data.metrics.http_req_duration.values['p(99)']}ms\n`;
  summary += indent + `  Max: ${data.metrics.http_req_duration.values.max}ms\n\n`;
  
  // Custom metrics
  if (data.metrics.cache_hits) {
    summary += indent + 'Cache Metrics:\n';
    summary += indent + `  Hit Rate: ${(data.metrics.cache_hits.values.rate * 100).toFixed(2)}%\n\n`;
  }
  
  // Thresholds
  summary += indent + 'Thresholds:\n';
  Object.keys(data.metrics).forEach(metric => {
    if (data.metrics[metric].thresholds) {
      Object.keys(data.metrics[metric].thresholds).forEach(threshold => {
        const passed = data.metrics[metric].thresholds[threshold].ok;
        const status = passed ? '✅ PASS' : '❌ FAIL';
        summary += indent + `  ${status}: ${metric} ${threshold}\n`;
      });
    }
  });
  
  summary += '\n' + indent + '='.repeat(80) + '\n';
  
  return summary;
}
