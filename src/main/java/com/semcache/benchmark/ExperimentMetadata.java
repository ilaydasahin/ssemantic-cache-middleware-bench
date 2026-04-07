package com.semcache.benchmark;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.Map;

/**
 * Extended metadata for experiment reproducibility and provenance tracking.
 * 
 * Captures:
 * - System information (OS, CPU, memory)
 * - Software versions (Java, dependencies)
 * - Git commit hash (if available)
 * - Dataset fingerprint (SHA-256)
 * - Configuration parameters
 * 
 * This enables:
 * - Independent verification
 * - Debugging performance anomalies
 * - Tracking configuration drift
 * - Reproducibility audits
 */
public record ExperimentMetadata(
    // Experiment identification
    @JsonProperty("experimentId") String experimentId,
    @JsonProperty("timestamp") String timestamp,
    @JsonProperty("durationSeconds") long durationSeconds,
    
    // System information
    @JsonProperty("system") SystemInfo system,
    
    // Software versions
    @JsonProperty("software") SoftwareInfo software,
    
    // Dataset provenance
    @JsonProperty("dataset") DatasetInfo dataset,
    
    // Configuration snapshot
    @JsonProperty("configuration") Map<String, Object> configuration,
    
    // Git provenance (if available)
    @JsonProperty("git") GitInfo git
) {
    
    public record SystemInfo(
        @JsonProperty("os") String os,
        @JsonProperty("osVersion") String osVersion,
        @JsonProperty("arch") String arch,
        @JsonProperty("cpuCores") int cpuCores,
        @JsonProperty("totalMemoryMB") long totalMemoryMB,
        @JsonProperty("maxHeapMB") long maxHeapMB
    ) {
        public static SystemInfo capture() {
            Runtime runtime = Runtime.getRuntime();
            return new SystemInfo(
                System.getProperty("os.name"),
                System.getProperty("os.version"),
                System.getProperty("os.arch"),
                runtime.availableProcessors(),
                getTotalPhysicalMemoryMB(),
                runtime.maxMemory() / (1024 * 1024)
            );
        }
        
        private static long getTotalPhysicalMemoryMB() {
            try {
                com.sun.management.OperatingSystemMXBean osBean = 
                    (com.sun.management.OperatingSystemMXBean) 
                    java.lang.management.ManagementFactory.getOperatingSystemMXBean();
                // Use getTotalMemorySize() instead of deprecated getTotalPhysicalMemorySize()
                return osBean.getTotalMemorySize() / (1024 * 1024);
            } catch (Exception e) {
                return -1; // Unknown
            }
        }
    }
    
    public record SoftwareInfo(
        @JsonProperty("javaVersion") String javaVersion,
        @JsonProperty("javaVendor") String javaVendor,
        @JsonProperty("springBootVersion") String springBootVersion,
        @JsonProperty("onnxRuntimeVersion") String onnxRuntimeVersion
    ) {
        public static SoftwareInfo capture() {
            return new SoftwareInfo(
                System.getProperty("java.version"),
                System.getProperty("java.vendor"),
                getSpringBootVersion(),
                getOnnxRuntimeVersion()
            );
        }
        
        private static String getSpringBootVersion() {
            try {
                return org.springframework.boot.SpringBootVersion.getVersion();
            } catch (Exception e) {
                return "unknown";
            }
        }
        
        private static String getOnnxRuntimeVersion() {
            try {
                // OrtEnvironment.getVersion() is an instance method, need to create environment
                try (ai.onnxruntime.OrtEnvironment env = ai.onnxruntime.OrtEnvironment.getEnvironment()) {
                    return env.getVersion();
                }
            } catch (Exception e) {
                return "unknown";
            }
        }
    }
    
    public record DatasetInfo(
        @JsonProperty("name") String name,
        @JsonProperty("path") String path,
        @JsonProperty("size") long size,
        @JsonProperty("lineCount") int lineCount,
        @JsonProperty("sha256") String sha256
    ) {}
    
    public record GitInfo(
        @JsonProperty("commitHash") String commitHash,
        @JsonProperty("branch") String branch,
        @JsonProperty("isDirty") boolean isDirty,
        @JsonProperty("remoteUrl") String remoteUrl
    ) {
        public static GitInfo capture() {
            try {
                // Try to read git info from .git directory
                java.io.File gitDir = new java.io.File(".git");
                if (!gitDir.exists()) {
                    return null;
                }
                
                // Read HEAD to get current commit
                String commitHash = readGitHead();
                String branch = readGitBranch();
                boolean isDirty = checkGitDirty();
                String remoteUrl = readGitRemote();
                
                return new GitInfo(commitHash, branch, isDirty, remoteUrl);
            } catch (Exception e) {
                return null; // Git info not available
            }
        }
        
        private static String readGitHead() {
            try {
                java.nio.file.Path headPath = java.nio.file.Paths.get(".git/HEAD");
                String head = java.nio.file.Files.readString(headPath).trim();
                
                if (head.startsWith("ref: ")) {
                    // Read the ref file
                    String refPath = head.substring(5);
                    java.nio.file.Path refFile = java.nio.file.Paths.get(".git/" + refPath);
                    return java.nio.file.Files.readString(refFile).trim();
                } else {
                    // Direct commit hash
                    return head;
                }
            } catch (Exception e) {
                return "unknown";
            }
        }
        
        private static String readGitBranch() {
            try {
                java.nio.file.Path headPath = java.nio.file.Paths.get(".git/HEAD");
                String head = java.nio.file.Files.readString(headPath).trim();
                
                if (head.startsWith("ref: refs/heads/")) {
                    return head.substring(16);
                }
                return "detached";
            } catch (Exception e) {
                return "unknown";
            }
        }
        
        private static boolean checkGitDirty() {
            try {
                // Simple check: if .git/index exists and is newer than HEAD
                java.io.File index = new java.io.File(".git/index");
                java.io.File head = new java.io.File(".git/HEAD");
                
                if (!index.exists() || !head.exists()) {
                    return false;
                }
                
                return index.lastModified() > head.lastModified();
            } catch (Exception e) {
                return false;
            }
        }
        
        private static String readGitRemote() {
            try {
                java.nio.file.Path configPath = java.nio.file.Paths.get(".git/config");
                String config = java.nio.file.Files.readString(configPath);
                
                // Simple regex to extract remote URL
                java.util.regex.Pattern pattern = java.util.regex.Pattern.compile(
                    "\\[remote \"origin\"\\].*?url = (.+?)\\n", 
                    java.util.regex.Pattern.DOTALL
                );
                java.util.regex.Matcher matcher = pattern.matcher(config);
                
                if (matcher.find()) {
                    return matcher.group(1).trim();
                }
                return "unknown";
            } catch (Exception e) {
                return "unknown";
            }
        }
    }
    
    /**
     * Creates metadata snapshot at experiment start.
     */
    public static ExperimentMetadata create(ExperimentConfig config) {
        return new ExperimentMetadata(
            config.experimentId(),
            Instant.now().toString(),
            0, // Will be updated at end
            SystemInfo.capture(),
            SoftwareInfo.capture(),
            null, // Will be set after dataset loading
            configToMap(config),
            GitInfo.capture()
        );
    }
    
    /**
     * Updates metadata with dataset information after loading.
     */
    public ExperimentMetadata withDatasetInfo(DatasetInfo datasetInfo) {
        return new ExperimentMetadata(
            experimentId,
            timestamp,
            durationSeconds,
            system,
            software,
            datasetInfo,
            configuration,
            git
        );
    }
    
    /**
     * Updates metadata with final duration.
     */
    public ExperimentMetadata withDuration(long durationSeconds) {
        return new ExperimentMetadata(
            experimentId,
            timestamp,
            durationSeconds,
            system,
            software,
            dataset,
            configuration,
            git
        );
    }
    
    private static Map<String, Object> configToMap(ExperimentConfig config) {
        return Map.ofEntries(
            Map.entry("datasetName", config.datasetName()),
            Map.entry("embeddingModel", config.embeddingModelName()),
            Map.entry("similarityThreshold", config.similarityThreshold()),
            Map.entry("warmupStrategy", config.warmupStrategy()),
            Map.entry("warmupRatio", config.warmupRatio()),
            Map.entry("randomSeed", config.randomSeed()),
            Map.entry("sampleSize", config.sampleSize()),
            Map.entry("hnswEnabled", config.hnswEnabled()),
            Map.entry("cacheStrategy", config.cacheStrategy()),
            Map.entry("knnK", config.knnK()),
            Map.entry("maxCacheEntries", config.maxCacheEntries()),
            Map.entry("ttlSeconds", config.ttlSeconds()),
            Map.entry("zipfianSkew", config.zipfianSkew()),
            Map.entry("noiseProbability", config.noiseProbability())
        );
    }
}
