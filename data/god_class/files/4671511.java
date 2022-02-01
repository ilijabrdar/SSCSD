@Deprecated
@InterfaceAudience.Public
@InterfaceStability.Stable
public class Job extends ControlledJob {
  static final Log LOG = LogFactory.getLog(Job.class);

  final public static int SUCCESS = 0;
  final public static int WAITING = 1;
  final public static int RUNNING = 2;
  final public static int READY = 3;
  final public static int FAILED = 4;
  final public static int DEPENDENT_FAILED = 5;

  /** 
   * Construct a job.
   * @param jobConf a mapred job configuration representing a job to be executed.
   * @param dependingJobs an array of jobs the current job depends on
   */
  @SuppressWarnings("unchecked")
  public Job(JobConf jobConf, ArrayList<?> dependingJobs) throws IOException {
    super(new org.apache.hadoop.mapreduce.Job(jobConf), 
          (List<ControlledJob>) dependingJobs);
  }

  public Job(JobConf conf) throws IOException {
    super(conf);
  }

  /**
   * @return the mapred ID of this job as assigned by the 
   * mapred framework.
   */
  public JobID getAssignedJobID() {
    org.apache.hadoop.mapreduce.JobID temp = super.getMapredJobID();
    if(temp == null) {
      return null;
    }
    return JobID.downgrade(temp);
  }

  /**
   * @deprecated setAssignedJobID should not be called.
   * JOBID is set by the framework.
   */
  @Deprecated
  public void setAssignedJobID(JobID mapredJobID) {
    // do nothing
  }

  /**
   * @return the mapred job conf of this job
   */
  public synchronized JobConf getJobConf() {
    return new JobConf(super.getJob().getConfiguration());
  }


  /**
   * Set the mapred job conf for this job.
   * @param jobConf the mapred job conf for this job.
   */
  public synchronized void setJobConf(JobConf jobConf) {
    try {
      super.setJob(new org.apache.hadoop.mapreduce.Job(jobConf));
    } catch (IOException ioe) { 
      LOG.info("Exception" + ioe);
    }
  }

  /**
   * @return the state of this job
   */
  public synchronized int getState() {
    State state = super.getJobState();
    if (state == State.SUCCESS) {
      return SUCCESS;
    } 
    if (state == State.WAITING) {
      return WAITING;
    }
    if (state == State.RUNNING) {
      return RUNNING;
    }
    if (state == State.READY) {
      return READY;
    }
    if (state == State.FAILED ) {
      return FAILED;
    }
    if (state == State.DEPENDENT_FAILED ) {
      return DEPENDENT_FAILED;
    }
    return -1;
  }
  
  /**
   * @return the job client of this job
   */
  public JobClient getJobClient() {
    try {
      return new JobClient(super.getJob().getConfiguration());
    } catch (IOException ioe) {
      return null;
    }
  }

  /**
   * @return the depending jobs of this job
   */
  public ArrayList<Job> getDependingJobs() {
    return JobControl.castToJobList(super.getDependentJobs());
  }

}