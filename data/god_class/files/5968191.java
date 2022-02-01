public class OutputScanner extends FilterOutputStream {

    private final CountDownLatch found = new CountDownLatch(1);

    public OutputScanner(final OutputStream out, final String scanString) {
        super(null);
        this.out = new Scan(out, scanString);
    }

    public class Scan extends FilterOutputStream {

        private final ScanBuffer scan;

        public Scan(final OutputStream out, final String scanString) {
            super(out);
            scan = new ScanBuffer(scanString);
        }

        @Override
        public void write(final int b) throws IOException {
            check(b);
            super.write(b);
        }

        @Override
        public void write(final byte[] bytes) throws IOException {
            for (final byte b : bytes) {
                check(b);
            }
            super.write(bytes);
        }

        @Override
        public void write(final byte[] bytes, final int off, final int len) throws IOException {
            for (int i = off; i < len; i++) {
                check(bytes[i]);
            }
            super.write(bytes, off, len);
        }

        private void check(final int b) {
            scan.append(b);
            if (scan.match()) {
                // Cut ourselves out of the call chain.
                //
                // This works because
                //  - ScanningOutputStreamFilter.this.out == this
                //  - this.out != this)
                //
                // Our parent is delegating to us and we are delegating
                // to the actual OutputStream
                //
                // To cut ourselves out of the call chain and eliminate
                // the overhead of checking the ScanBuffer, we set our
                // parent to not delegate to us and to instead delegate
                // to the actual OutputStream.

                // Intellij mistakenly shows this grayed out,
                // however it is very very significant.
                OutputScanner.this.out = this.out;
                found.countDown();
            }
        }
    }


    public void await() throws InterruptedException {
        found.await();
    }

    public boolean await(final long timeout, final TimeUnit unit) throws InterruptedException {
        return found.await(timeout, unit);
    }
}