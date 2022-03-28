package extractor;

import com.beust.jcommander.Parameter;

public class Args {
    @Parameter(names = "-file")
    private String filePath;

    @Parameter(names = "-output")
    private String outputDir;

    public String getFilePath() {
        return filePath;
    }

    public String getOutputDir() {
        return outputDir;
    }
}
