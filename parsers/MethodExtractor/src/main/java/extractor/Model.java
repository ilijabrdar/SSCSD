package extractor;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Getter
@Setter
@AllArgsConstructor
public class Model {
    private String filePath;
    private String name;
    private int paramNumber;
    private List<String> paramTypes;
    private String code;
}
