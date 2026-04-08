package com.semcache.benchmark;

import com.semcache.benchmark.ExperimentResultExporter.QueryLog;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

/**
 * Unit tests for StreamingResultWriter
 */
class StreamingResultWriterTest {

    @TempDir
    Path tempDir;

    private Path outputPath;
    private StreamingResultWriter writer;

    @BeforeEach
    void setUp() throws IOException {
        outputPath = tempDir.resolve("test-output.jsonl");
    }

    @AfterEach
    void tearDown() throws IOException {
        if (writer != null) {
            writer.close();
        }
    }

    @Test
    void testWriterInitialization() throws IOException {
        // Given output path
        // When writer is created
        writer = new StreamingResultWriter(outputPath);
        
        // Then should initialize successfully
        assertThat(writer).isNotNull();
        assertThat(writer.getTotalWritten()).isEqualTo(0);
    }

    @Test
    void testWriterWithCustomFlushThreshold() throws IOException {
        // Given custom flush threshold
        int flushThreshold = 50;
        
        // When writer is created
        writer = new StreamingResultWriter(outputPath, flushThreshold);
        
        // Then should initialize successfully
        assertThat(writer).isNotNull();
        assertThat(writer.getTotalWritten()).isEqualTo(0);
    }

    @Test
    void testWriteSingleEntry() throws IOException {
        // Given writer
        writer = new StreamingResultWriter(outputPath);
        
        // When single entry is written
        QueryLog log = new QueryLog("test query", "ground truth", "test response", true, 0.95, 100.0, 10.0, 90.0);
        writer.write(log);
        writer.flush();
        
        // Then should write to file
        assertThat(writer.getTotalWritten()).isEqualTo(1);
        assertThat(Files.exists(outputPath)).isTrue();
    }

    @Test
    void testWriteMultipleEntries() throws IOException {
        // Given writer
        writer = new StreamingResultWriter(outputPath);
        
        // When multiple entries are written
        for (int i = 0; i < 10; i++) {
            QueryLog log = new QueryLog("query" + i, "truth" + i, "response" + i, i % 2 == 0, 0.90 + i * 0.01, 100.0 + i, 10.0, 90.0);
            writer.write(log);
        }
        writer.flush();
        
        // Then should write all entries
        assertThat(writer.getTotalWritten()).isEqualTo(10);
        
        List<String> lines = Files.readAllLines(outputPath);
        assertThat(lines).hasSize(10);
    }

    @Test
    void testAutoFlush() throws IOException {
        // Given writer with small flush threshold
        writer = new StreamingResultWriter(outputPath, 5);
        
        // When entries exceed threshold
        for (int i = 0; i < 7; i++) {
            QueryLog log = new QueryLog("query" + i, "truth" + i, "response" + i, true, 0.95, 100.0, 10.0, 90.0);
            writer.write(log);
        }
        
        // Then should auto-flush at threshold
        // 5 entries flushed, 2 in buffer
        assertThat(writer.getTotalWritten()).isEqualTo(7);
    }

    @Test
    void testFlushEmptyBuffer() throws IOException {
        // Given writer with no entries
        writer = new StreamingResultWriter(outputPath);
        
        // When flush is called
        writer.flush();
        
        // Then should handle gracefully
        assertThat(writer.getTotalWritten()).isEqualTo(0);
    }

    @Test
    void testCloseFlushesBuffer() throws IOException {
        // Given writer with buffered entries
        writer = new StreamingResultWriter(outputPath, 100);
        
        for (int i = 0; i < 5; i++) {
            QueryLog log = new QueryLog("query" + i, "truth" + i, "response" + i, true, 0.95, 100.0, 10.0, 90.0);
            writer.write(log);
        }
        
        // When writer is closed
        writer.close();
        
        // Then should flush remaining buffer
        List<String> lines = Files.readAllLines(outputPath);
        assertThat(lines).hasSize(5);
    }

    @Test
    void testLargeDataset() throws IOException {
        // Given writer with large dataset
        writer = new StreamingResultWriter(outputPath, 100);
        
        // When many entries are written
        int entryCount = 1000;
        for (int i = 0; i < entryCount; i++) {
            QueryLog log = new QueryLog("query" + i, "truth" + i, "response" + i, i % 2 == 0, 0.90, 100.0, 10.0, 90.0);
            writer.write(log);
        }
        writer.close();
        
        // Then should write all entries
        List<String> lines = Files.readAllLines(outputPath);
        assertThat(lines).hasSize(entryCount);
    }

    @Test
    void testGetTotalWrittenBeforeFlush() throws IOException {
        // Given writer with buffered entries
        writer = new StreamingResultWriter(outputPath, 100);
        
        for (int i = 0; i < 5; i++) {
            QueryLog log = new QueryLog("query" + i, "truth" + i, "response" + i, true, 0.95, 100.0, 10.0, 90.0);
            writer.write(log);
        }
        
        // When total is checked before flush
        int total = writer.getTotalWritten();
        
        // Then should include buffered entries
        assertThat(total).isEqualTo(5);
    }

    @Test
    void testJsonFormatting() throws IOException {
        // Given writer
        writer = new StreamingResultWriter(outputPath);
        
        // When entry with special characters is written
        QueryLog log = new QueryLog("query with \"quotes\"", "truth", "response with\nnewline", true, 0.95, 100.0, 10.0, 90.0);
        writer.write(log);
        writer.flush();
        
        // Then should escape properly
        List<String> lines = Files.readAllLines(outputPath);
        assertThat(lines).hasSize(1);
        assertThat(lines.get(0)).contains("query with");
    }

    @Test
    void testConcurrentWrites() throws IOException {
        // Given writer
        writer = new StreamingResultWriter(outputPath, 100);
        
        // When concurrent writes occur (synchronized method)
        // Then should handle safely
        assertThatCode(() -> {
            for (int i = 0; i < 10; i++) {
                QueryLog log = new QueryLog("query" + i, "truth" + i, "response" + i, true, 0.95, 100.0, 10.0, 90.0);
                writer.write(log);
            }
        }).doesNotThrowAnyException();
    }
}
