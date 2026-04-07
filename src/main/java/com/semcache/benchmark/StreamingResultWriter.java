package com.semcache.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.semcache.benchmark.ExperimentResultExporter.QueryLog;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Streaming writer for query logs to prevent memory exhaustion.
 * 
 * Instead of accumulating all query logs in memory, this writer:
 * 1. Buffers logs in chunks (default: 100 entries)
 * 2. Flushes to disk periodically
 * 3. Prevents OutOfMemoryError on large experiments (100K+ queries)
 * 
 * Memory savings:
 * - Old approach: 100K queries × 1KB/query = 100 MB in memory
 * - New approach: 100 queries × 1KB/query = 100 KB in memory (1000× reduction)
 */
public class StreamingResultWriter implements AutoCloseable {
    
    private static final Logger log = LoggerFactory.getLogger(StreamingResultWriter.class);
    
    private final BufferedWriter writer;
    private final ObjectMapper objectMapper;
    private final int flushThreshold;
    private final List<QueryLog> buffer;
    private int totalWritten = 0;
    
    public StreamingResultWriter(Path outputPath, int flushThreshold) throws IOException {
        this.writer = new BufferedWriter(new FileWriter(outputPath.toFile()));
        this.objectMapper = new ObjectMapper();
        this.flushThreshold = flushThreshold;
        this.buffer = new ArrayList<>(flushThreshold);
        
        log.info("Streaming writer initialized: output={}, flushThreshold={}", 
                outputPath, flushThreshold);
    }
    
    public StreamingResultWriter(Path outputPath) throws IOException {
        this(outputPath, 100);
    }
    
    /**
     * Adds a query log entry to the buffer.
     * Automatically flushes when buffer reaches threshold.
     */
    public synchronized void write(QueryLog log) throws IOException {
        buffer.add(log);
        
        if (buffer.size() >= flushThreshold) {
            flush();
        }
    }
    
    /**
     * Flushes buffered entries to disk.
     */
    public synchronized void flush() throws IOException {
        if (buffer.isEmpty()) {
            return;
        }
        
        for (QueryLog log : buffer) {
            String json = objectMapper.writeValueAsString(log);
            writer.write(json);
            writer.newLine();
        }
        
        writer.flush();
        totalWritten += buffer.size();
        
        log.debug("Flushed {} entries to disk (total: {})", buffer.size(), totalWritten);
        buffer.clear();
    }
    
    /**
     * Gets total number of entries written.
     */
    public int getTotalWritten() {
        return totalWritten + buffer.size();
    }
    
    @Override
    public void close() throws IOException {
        try {
            flush(); // Flush remaining buffer
            writer.close();
            log.info("Streaming writer closed: total entries written = {}", totalWritten);
        } catch (IOException e) {
            log.error("Error closing streaming writer", e);
            throw e;
        }
    }
}
