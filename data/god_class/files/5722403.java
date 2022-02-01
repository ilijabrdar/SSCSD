public class SequenceFileReader<Key extends Writable, Value extends Writable>
    extends AbstractFileReader {
    public static final String[] defaultFields = { "key", "value" };
    public static final String BUFFER_SIZE = "hdfsspout.reader.buffer.bytes";
    private static final Logger LOG = LoggerFactory
        .getLogger(SequenceFileReader.class);
    private static final int DEFAULT_BUFF_SIZE = 4096;
    private final SequenceFile.Reader reader;

    private final SequenceFileReader.Offset offset;


    private final Key key;
    private final Value value;


    public SequenceFileReader(FileSystem fs, Path file, Map<String, Object> conf)
        throws IOException {
        super(fs, file);
        int bufferSize = !conf.containsKey(BUFFER_SIZE) ? DEFAULT_BUFF_SIZE : Integer.parseInt(conf.get(BUFFER_SIZE).toString());
        this.reader = new SequenceFile.Reader(fs.getConf(), SequenceFile.Reader.file(file), SequenceFile.Reader.bufferSize(bufferSize));
        this.key = (Key) ReflectionUtils.newInstance(reader.getKeyClass(), fs.getConf());
        this.value = (Value) ReflectionUtils.newInstance(reader.getValueClass(), fs.getConf());
        this.offset = new SequenceFileReader.Offset(0, 0, 0);
    }

    public SequenceFileReader(FileSystem fs, Path file, Map<String, Object> conf, String offset)
        throws IOException {
        super(fs, file);
        int bufferSize = !conf.containsKey(BUFFER_SIZE) ? DEFAULT_BUFF_SIZE : Integer.parseInt(conf.get(BUFFER_SIZE).toString());
        this.offset = new SequenceFileReader.Offset(offset);
        this.reader = new SequenceFile.Reader(fs.getConf(), SequenceFile.Reader.file(file), SequenceFile.Reader.bufferSize(bufferSize));
        this.key = (Key) ReflectionUtils.newInstance(reader.getKeyClass(), fs.getConf());
        this.value = (Value) ReflectionUtils.newInstance(reader.getValueClass(), fs.getConf());
        skipToOffset(this.reader, this.offset, this.key);
    }

    private static <K> void skipToOffset(SequenceFile.Reader reader, Offset offset, K key) throws IOException {
        reader.sync(offset.lastSyncPoint);
        for (int i = 0; i < offset.recordsSinceLastSync; ++i) {
            reader.next(key);
        }
    }

    public List<Object> next() throws IOException, ParseException {
        if (reader.next(key, value)) {
            ArrayList<Object> result = new ArrayList<Object>(2);
            Collections.addAll(result, key, value);
            offset.increment(reader.syncSeen(), reader.getPosition());
            return result;
        }
        return null;
    }

    @Override
    public void close() {
        try {
            reader.close();
        } catch (IOException e) {
            LOG.warn("Ignoring error when closing file " + getFilePath(), e);
        }
    }

    public Offset getFileOffset() {
        return offset;
    }


    public static class Offset implements FileOffset {
        public long lastSyncPoint;
        public long recordsSinceLastSync;
        public long currentRecord;
        private long currRecordEndOffset;
        private long prevRecordEndOffset;

        public Offset(long lastSyncPoint, long recordsSinceLastSync, long currentRecord) {
            this(lastSyncPoint, recordsSinceLastSync, currentRecord, 0, 0);
        }

        public Offset(long lastSyncPoint, long recordsSinceLastSync, long currentRecord
            , long currRecordEndOffset, long prevRecordEndOffset) {
            this.lastSyncPoint = lastSyncPoint;
            this.recordsSinceLastSync = recordsSinceLastSync;
            this.currentRecord = currentRecord;
            this.prevRecordEndOffset = prevRecordEndOffset;
            this.currRecordEndOffset = currRecordEndOffset;
        }

        public Offset(String offset) {
            try {
                if (offset == null) {
                    throw new IllegalArgumentException("offset cannot be null");
                }
                if (offset.equalsIgnoreCase("0")) {
                    this.lastSyncPoint = 0;
                    this.recordsSinceLastSync = 0;
                    this.currentRecord = 0;
                    this.prevRecordEndOffset = 0;
                    this.currRecordEndOffset = 0;
                } else {
                    String[] parts = offset.split(":");
                    this.lastSyncPoint = Long.parseLong(parts[0].split("=")[1]);
                    this.recordsSinceLastSync = Long.parseLong(parts[1].split("=")[1]);
                    this.currentRecord = Long.parseLong(parts[2].split("=")[1]);
                    this.prevRecordEndOffset = 0;
                    this.currRecordEndOffset = 0;
                }
            } catch (Exception e) {
                throw new IllegalArgumentException("'" + offset +
                                                   "' cannot be interpreted. It is not in expected format for SequenceFileReader." +
                                                   " Format e.g. {sync=123:afterSync=345:record=67}");
            }
        }

        @Override
        public String toString() {
            return '{' +
                   "sync=" + lastSyncPoint +
                   ":afterSync=" + recordsSinceLastSync +
                   ":record=" + currentRecord +
                   ":}";
        }

        @Override
        public boolean isNextOffset(FileOffset rhs) {
            if (rhs instanceof Offset) {
                Offset other = ((Offset) rhs);
                return other.currentRecord > currentRecord + 1;
            }
            return false;
        }

        @Override
        public int compareTo(FileOffset o) {
            Offset rhs = ((Offset) o);
            if (currentRecord < rhs.currentRecord) {
                return -1;
            }
            if (currentRecord == rhs.currentRecord) {
                return 0;
            }
            return 1;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (!(o instanceof Offset)) {
                return false;
            }

            Offset offset = (Offset) o;

            return currentRecord == offset.currentRecord;
        }

        @Override
        public int hashCode() {
            return (int) (currentRecord ^ (currentRecord >>> 32));
        }

        void increment(boolean syncSeen, long newBytePosition) {
            if (!syncSeen) {
                ++recordsSinceLastSync;
            } else {
                recordsSinceLastSync = 1;
                lastSyncPoint = prevRecordEndOffset;
            }
            ++currentRecord;
            prevRecordEndOffset = currRecordEndOffset;
            currentRecord = newBytePosition;
        }

        @Override
        public Offset clone() {
            return new Offset(lastSyncPoint, recordsSinceLastSync, currentRecord, currRecordEndOffset, prevRecordEndOffset);
        }

    } //class Offset
} //class