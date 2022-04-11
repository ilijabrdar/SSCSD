public DataSet<PieEntry> copy(){
    List<PieEntry> entries = new ArrayList<>();
    for (int i = 0; i < mEntries.size(); i++) {
        entries.add(mEntries.get(i).copy());
    }
    PieDataSet copied = new PieDataSet(entries, getLabel());
    copy(copied);
    return copied;
}