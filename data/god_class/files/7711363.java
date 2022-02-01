public class ControllerLiveInstanceChangeListener implements LiveInstanceChangeListener {

  private static final Logger LOGGER = LoggerFactory.getLogger(ControllerLiveInstanceChangeListener.class);

  private final ControllerHelixManager _controllerHelixManager;
  private final HelixManager _helixManager;

  private final int minIntervalInSeconds = 60;
  private long _lastRebalanceTimeMillis = 0;

  private final ScheduledExecutorService _delayedScheduler = Executors.newSingleThreadScheduledExecutor();

  private final Counter _isLeaderCounter = new Counter();

  public ControllerLiveInstanceChangeListener(ControllerHelixManager controllerHelixManager,
      HelixManager helixManager, int _workloadRefreshPeriodInSeconds) {
    _controllerHelixManager = controllerHelixManager;
    _helixManager = helixManager;
    registerMetrics();

    LOGGER.info("Trying to schedule auto rebalancing");
    _delayedScheduler.scheduleWithFixedDelay(
        new Runnable() {
          @Override
          public void run() {
            try {
              rebalanceCurrentCluster(false);
            } catch (Exception e) {
              LOGGER.error("Got exception during periodically rebalancing the whole cluster! ", e);
            }
          }
        }, 60, _workloadRefreshPeriodInSeconds, TimeUnit.SECONDS);
  }

  @Override
  public void onLiveInstanceChange(final List<LiveInstance> liveInstances, NotificationContext changeContext) {
    LOGGER.info("ControllerLiveInstanceChangeListener.onLiveInstanceChange() wakes up!");
    _delayedScheduler.schedule(new Runnable() {
      @Override
      public void run() {
        try {
          rebalanceCurrentCluster(true);
        } catch (Exception e) {
          LOGGER.error("Got exception during rebalance the whole cluster! ", e);
        }
      }
    }, 5, TimeUnit.SECONDS);
  }

  public synchronized void rebalanceCurrentCluster(boolean onlyCheckOffline) {
    if (_helixManager.isLeader()) {
      _isLeaderCounter.inc(1 - _isLeaderCounter.getCount());
    } else {
      LOGGER.info("Not leader, do nothing!");
      return;
    }

    if (HelixUtils.liveInstances(_helixManager).isEmpty() ||
        HelixUtils.liveInstances(_controllerHelixManager.getWorkerHelixManager().getHelixManager()).isEmpty()) {
      LOGGER.info("No live instances, do nothing!");
      return;
    }

    LOGGER.info("onlyCheckOffline={}", onlyCheckOffline);
    try {
      _controllerHelixManager.handleLiveInstanceChange(onlyCheckOffline, false);
    } catch (Exception e) {
      LOGGER.error("Failed to handle live instance change!", e);
    }

  }

  private void registerMetrics() {
    try {
      HelixKafkaMirrorMakerMetricsReporter.get().registerMetric("leader.counter",
          _isLeaderCounter);
    } catch (Exception e) {
      LOGGER.error("Error registering metrics!", e);
    }
  }

}