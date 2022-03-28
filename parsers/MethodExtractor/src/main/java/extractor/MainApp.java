package extractor;

import com.beust.jcommander.JCommander;
import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.resolution.declarations.ResolvedReferenceTypeDeclaration;
import com.github.javaparser.resolution.types.ResolvedType;
import com.github.javaparser.symbolsolver.javaparsermodel.JavaParserFacade;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.*;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

public class MainApp {
    public static void main(String[] args) throws FileNotFoundException, ClassNotFoundException {
//        String path = "C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\unlabeled\\Mindustry-master\\core\\src\\mindustry\\graphics\\g3d\\PlanetRenderer.java";
//        String outputPath = "";
        Args arguments = new Args();
        JCommander.newBuilder().addObject(arguments).build().parse(args);
        Gson converter = new GsonBuilder().setPrettyPrinting().create();

        CombinedTypeSolver combinedTypeSolver = new CombinedTypeSolver();
        combinedTypeSolver.add(new ReflectionTypeSolver());
        combinedTypeSolver.add(new JavaParserTypeSolver(new File("C:\\Users\\ilija\\OneDrive\\Desktop\\SIAP\\data\\unlabeled\\Mindustry-master\\core\\src\\mindustry\\graphics\\g3d")));

        CompilationUnit unit = StaticJavaParser.parse(new File(arguments.getFilePath()));
        ClassOrInterfaceDeclaration klass = unit.findAll(ClassOrInterfaceDeclaration.class)
                .stream().findFirst().orElseThrow(ClassNotFoundException::new);

        List<MethodDeclaration> methods = klass.getMethods();
        methods.forEach(method -> {
            List<String> paramTypes = new ArrayList<>();
            List<Parameter> params = method.getParameters();
            params.forEach(param -> paramTypes.add(param.getType().asString()));
            String code;
            if(method.getBody().isPresent())
                code = method.getBody().get().toString();
            else
                throw new NullPointerException();
            code = method.getDeclarationAsString() + code;
            Model model = new Model(arguments.getFilePath(),
                    method.getNameAsString(), params.size(), paramTypes, code);
            String json = converter.toJson(model);
            try (PrintWriter out = new PrintWriter(new OutputStreamWriter(
                    new FileOutputStream(arguments.getOutputDir() + File.separator + UUID.randomUUID() + ".java")))) {
                out.print(json);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        });
    }

}

