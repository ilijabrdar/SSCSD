public class ConnectedOutParaManager extends IOutParaManager {

	public ConnectedOutParaManager(Client client) {
		super(client);
	}

	@Override
	public void register(OutPara para) {
		/*
		 * AC
		 * ACACAC
		 * 
		 */
		if (null != para && null != para.getKey() && !contains(para.getKey()))
		{
			para.setClient(client.getKey());
			outParaMap.put(para.getKey(), para);

			// ACUI
			OpUIManager.addItemToAC(para);
		}
	}

}