@Named
public class DefaultUpdatePolicyAnalyzer
    implements UpdatePolicyAnalyzer
{

    private static final Logger LOGGER = LoggerFactory.getLogger( DefaultUpdatePolicyAnalyzer.class );

    public DefaultUpdatePolicyAnalyzer()
    {
        // enables default constructor
    }

    public String getEffectiveUpdatePolicy( RepositorySystemSession session, String policy1, String policy2 )
    {
        return ordinalOfUpdatePolicy( policy1 ) < ordinalOfUpdatePolicy( policy2 ) ? policy1 : policy2;
    }

    private int ordinalOfUpdatePolicy( String policy )
    {
        if ( RepositoryPolicy.UPDATE_POLICY_DAILY.equals( policy ) )
        {
            return 1440;
        }
        else if ( RepositoryPolicy.UPDATE_POLICY_ALWAYS.equals( policy ) )
        {
            return 0;
        }
        else if ( policy != null && policy.startsWith( RepositoryPolicy.UPDATE_POLICY_INTERVAL ) )
        {
            return getMinutes( policy );
        }
        else
        {
            // assume "never"
            return Integer.MAX_VALUE;
        }
    }

    public boolean isUpdatedRequired( RepositorySystemSession session, long lastModified, String policy )
    {
        boolean checkForUpdates;

        if ( policy == null )
        {
            policy = "";
        }

        if ( RepositoryPolicy.UPDATE_POLICY_ALWAYS.equals( policy ) )
        {
            checkForUpdates = true;
        }
        else if ( RepositoryPolicy.UPDATE_POLICY_DAILY.equals( policy ) )
        {
            Calendar cal = Calendar.getInstance();
            cal.set( Calendar.HOUR_OF_DAY, 0 );
            cal.set( Calendar.MINUTE, 0 );
            cal.set( Calendar.SECOND, 0 );
            cal.set( Calendar.MILLISECOND, 0 );

            checkForUpdates = cal.getTimeInMillis() > lastModified;
        }
        else if ( policy.startsWith( RepositoryPolicy.UPDATE_POLICY_INTERVAL ) )
        {
            int minutes = getMinutes( policy );

            Calendar cal = Calendar.getInstance();
            cal.add( Calendar.MINUTE, -minutes );

            checkForUpdates = cal.getTimeInMillis() > lastModified;
        }
        else
        {
            // assume "never"
            checkForUpdates = false;

            if ( !RepositoryPolicy.UPDATE_POLICY_NEVER.equals( policy ) )
            {
                LOGGER.warn( "Unknown repository update policy '{}', assuming '{}'",
                        policy, RepositoryPolicy.UPDATE_POLICY_NEVER );
            }
        }

        return checkForUpdates;
    }

    private int getMinutes( String policy )
    {
        int minutes;
        try
        {
            String s = policy.substring( RepositoryPolicy.UPDATE_POLICY_INTERVAL.length() + 1 );
            minutes = Integer.valueOf( s );
        }
        catch ( RuntimeException e )
        {
            minutes = 24 * 60;

            LOGGER.warn( "Non-parseable repository update policy '{}', assuming '{}:1440'",
                    policy, RepositoryPolicy.UPDATE_POLICY_INTERVAL );
        }
        return minutes;
    }

}