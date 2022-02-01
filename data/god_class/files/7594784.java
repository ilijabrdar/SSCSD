@Configuration
public class ResourceServerConfiguration extends WebSecurityConfigurerAdapter implements Ordered {

	private int order = 3;

	@Autowired(required = false)
	private TokenStore tokenStore;

	@Autowired(required = false)
	private AuthenticationEventPublisher eventPublisher;

	@Autowired(required = false)
	private Map<String, ResourceServerTokenServices> tokenServices;

	@Autowired
	private ApplicationContext context;

	private List<ResourceServerConfigurer> configurers = Collections.emptyList();

	@Autowired(required = false)
	private AuthorizationServerEndpointsConfiguration endpoints;

	@Override
	public int getOrder() {
		return order;
	}

	public void setOrder(int order) {
		this.order = order;
	}

	/**
	 * @param configurers the configurers to set
	 */
	@Autowired(required = false)
	public void setConfigurers(List<ResourceServerConfigurer> configurers) {
		this.configurers = configurers;
	}

	private static class NotOAuthRequestMatcher implements RequestMatcher {

		private FrameworkEndpointHandlerMapping mapping;

		public NotOAuthRequestMatcher(FrameworkEndpointHandlerMapping mapping) {
			this.mapping = mapping;
		}

		@Override
		public boolean matches(HttpServletRequest request) {
			String requestPath = getRequestPath(request);
			for (String path : mapping.getPaths()) {
				if (requestPath.startsWith(mapping.getPath(path))) {
					return false;
				}
			}
			return true;
		}

		private String getRequestPath(HttpServletRequest request) {
			String url = request.getServletPath();

			if (request.getPathInfo() != null) {
				url += request.getPathInfo();
			}

			return url;
		}

	}

	@Override
	protected void configure(HttpSecurity http) throws Exception {
		ResourceServerSecurityConfigurer resources = new ResourceServerSecurityConfigurer();
		ResourceServerTokenServices services = resolveTokenServices();
		if (services != null) {
			resources.tokenServices(services);
		}
		else {
			if (tokenStore != null) {
				resources.tokenStore(tokenStore);
			}
			else if (endpoints != null) {
				resources.tokenStore(endpoints.getEndpointsConfigurer().getTokenStore());
			}
		}
		if (eventPublisher != null) {
			resources.eventPublisher(eventPublisher);
		}
		for (ResourceServerConfigurer configurer : configurers) {
			configurer.configure(resources);
		}
		// @formatter:off
		http.authenticationProvider(new AnonymousAuthenticationProvider("default"))
		// N.B. exceptionHandling is duplicated in resources.configure() so that
		// it works
		.exceptionHandling()
				.accessDeniedHandler(resources.getAccessDeniedHandler()).and()
				.sessionManagement()
				.sessionCreationPolicy(SessionCreationPolicy.STATELESS).and()
				.csrf().disable();
		// @formatter:on
		http.apply(resources);
		if (endpoints != null) {
			// Assume we are in an Authorization Server
			http.requestMatcher(new NotOAuthRequestMatcher(endpoints.oauth2EndpointHandlerMapping()));
		}
		for (ResourceServerConfigurer configurer : configurers) {
			// Delegates can add authorizeRequests() here
			configurer.configure(http);
		}
		if (configurers.isEmpty()) {
			// Add anyRequest() last as a fall back. Spring Security would
			// replace an existing anyRequest() matcher with this one, so to
			// avoid that we only add it if the user hasn't configured anything.
			http.authorizeRequests().anyRequest().authenticated();
		}
	}

	private ResourceServerTokenServices resolveTokenServices() {
		if (tokenServices == null || tokenServices.size() == 0) {
			return null;
		}
		if (tokenServices.size() == 1) {
			return tokenServices.values().iterator().next();
		}
		if (tokenServices.size() == 2) {
			// Maybe they are the ones provided natively
			Iterator<ResourceServerTokenServices> iter = tokenServices.values().iterator();
			ResourceServerTokenServices one = iter.next();
			ResourceServerTokenServices two = iter.next();
			if (elementsEqual(one, two)) {
				return one;
			}
		}
		return context.getBean(ResourceServerTokenServices.class);
	}

	private boolean elementsEqual(Object one, Object two) {
		// They might just be equal
		if (one == two) {
			return true;
		}
		Object targetOne = findTarget(one);
		Object targetTwo = findTarget(two);
		return targetOne == targetTwo;
	}

	private Object findTarget(Object item) {
		Object current = item;
		while (current instanceof Advised) {
			try {
				current = ((Advised) current).getTargetSource().getTarget();
			}
			catch (Exception e) {
				ReflectionUtils.rethrowRuntimeException(e);
			}
		}
		return current;
	}

}