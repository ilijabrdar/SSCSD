public class AutomaticRowHeightExample extends AbstractNatExample {

    private static final Display DISPLAY = Display.getDefault();

    private final List<LogRecord> logMessages = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        StandaloneNatExampleRunner.run(new AutomaticRowHeightExample());
    }

    @Override
    public String getDescription() {
        return "Demonstrates how to implement a log viewer using NatTable with the percentage "
                + "sizing and the automatic row height calculation feature. If you resize the "
                + "window you will see the rows growing/shrinking to always show the whole content "
                + "by wrapping the text and resizing the row heights.";
    }

    @Override
    public Control createExampleControl(Composite parent) {
        Composite panel = new Composite(parent, SWT.NONE);

        GridData layoutData = GridDataFactory.fillDefaults().grab(true, true).create();

        GridLayout layout = new GridLayout();
        layout.numColumns = 1;
        layout.marginWidth = 0;
        layout.marginHeight = 0;
        panel.setLayout(layout);
        panel.setLayoutData(layoutData);

        loadMessages();

        ListDataProvider<LogRecord> dataProvider =
                new ListDataProvider<>(
                        this.logMessages,
                        new ReflectiveColumnPropertyAccessor<LogRecord>(new String[] { "message" })); //$NON-NLS-1$
        DataLayer dataLayer = new DataLayer(dataProvider);
        dataLayer.setColumnPercentageSizing(true);
        dataLayer.setColumnWidthPercentageByPosition(0, 100);
        dataLayer.setConfigLabelAccumulator(new ValidatorMessageLabelAccumulator());

        ViewportLayer layer = new ViewportLayer(dataLayer);
        layer.setRegionName(GridRegion.BODY);

        NatTable natTable = new NatTable(panel, NatTable.DEFAULT_STYLE_OPTIONS | SWT.BORDER, layer, false);
        natTable.addConfiguration(new ValidationMessageTableStyleConfiguration(parent));

        natTable.configure();

        GridDataFactory.fillDefaults().grab(true, true).applyTo(natTable);

        return panel;
    }

    private void loadMessages() {
        String text = "Lorem ipsum dolor sit amet, consetetur sadipscing elitr, "
                + "sed diam nonumy eirmod tempor invidunt ut labore et dolore "
                + "magna aliquyam erat, sed diam voluptua.";

        Level[] levels = new Level[] { Level.SEVERE, Level.WARNING, Level.INFO };
        String[] words = text.split(" ");

        Random levelRandom = new Random();
        Random wordRandom = new Random();
        for (int i = 0; i < 100; i++) {
            int randWords = wordRandom.nextInt(words.length);
            String msg = "";
            for (int j = 0; j < randWords; j++) {
                msg += words[j] + " ";
            }
            this.logMessages.add(
                    new LogRecord(levels[levelRandom.nextInt(levels.length)], msg));
        }
    }

    public class ValidatorMessageLabelAccumulator implements IConfigLabelAccumulator {
        @Override
        public void accumulateConfigLabels(LabelStack configLabels, int columnPosition, int rowPosition) {
            LogRecord vm = AutomaticRowHeightExample.this.logMessages.get(rowPosition);
            configLabels.addLabel(vm.getLevel().toString());
        }
    }

    public class ValidationMessageTableStyleConfiguration extends DefaultNatTableStyleConfiguration {

        private LocalResourceManager resManager;
        private Image errorImage;
        private Image warningImage;
        private Image infoImage;

        public ValidationMessageTableStyleConfiguration(Composite composite) {
            this.resManager = new LocalResourceManager(JFaceResources.getResources(), composite);

            int IMAGE_SIZE = GUIHelper.convertHorizontalPixelToDpi(16);

            this.errorImage = this.resManager.createImage(ImageDescriptor.createFromImageData(
                    DISPLAY.getSystemImage(SWT.ICON_ERROR).getImageData()
                            .scaledTo(IMAGE_SIZE, IMAGE_SIZE)));

            this.warningImage = this.resManager.createImage(
                    ImageDescriptor.createFromImageData(
                            DISPLAY.getSystemImage(SWT.ICON_WARNING).getImageData()
                                    .scaledTo(IMAGE_SIZE, IMAGE_SIZE)));

            this.infoImage = this.resManager.createImage(
                    ImageDescriptor.createFromImageData(
                            DISPLAY.getSystemImage(SWT.ICON_INFORMATION).getImageData()
                                    .scaledTo(IMAGE_SIZE, IMAGE_SIZE)));

            this.hAlign = HorizontalAlignmentEnum.LEFT;
            this.cellPainter = new LineBorderDecorator(
                    new PaddingDecorator(
                            new CellPainterDecorator(
                                    new AutomaticRowHeightTextPainter(2),
                                    CellEdgeEnum.LEFT, new ImagePainter()),
                            0, 2, 0, 2));
        }

        @Override
        public void configureRegistry(IConfigRegistry configRegistry) {
            super.configureRegistry(configRegistry);

            Style errorStyle = new Style();
            errorStyle.setAttributeValue(
                    CellStyleAttributes.IMAGE,
                    this.errorImage);
            configRegistry.registerConfigAttribute(
                    CellConfigAttributes.CELL_STYLE,
                    errorStyle,
                    DisplayMode.NORMAL,
                    Level.SEVERE.toString());

            Style warningStyle = new Style();
            warningStyle.setAttributeValue(
                    CellStyleAttributes.IMAGE,
                    this.warningImage);
            configRegistry.registerConfigAttribute(
                    CellConfigAttributes.CELL_STYLE,
                    warningStyle,
                    DisplayMode.NORMAL,
                    Level.WARNING.toString());

            Style informationStyle = new Style();
            informationStyle.setAttributeValue(
                    CellStyleAttributes.IMAGE,
                    this.infoImage);
            configRegistry.registerConfigAttribute(
                    CellConfigAttributes.CELL_STYLE,
                    informationStyle,
                    DisplayMode.NORMAL,
                    Level.INFO.toString());
        }
    }
}