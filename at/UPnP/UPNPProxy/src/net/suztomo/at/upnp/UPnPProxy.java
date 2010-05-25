package net.suztomo.at.upnp;
import java.util.HashMap;


public class UPnPProxy {
	private String serviceName_;
	private HashMap<String, UPnPService> services; /* String -> UPnPService */
	
	public UPnPProxy() {
		
	}
	
	public void setupProxyActor(Object o) {
		System.out.println("Setting up " + o);
	}
	
	
	
	
	/*
	 * Finds UPnP devices and services.
	 * Setups proxy actor for each service
	 */
	public void discover(String serviceName, ATUPnPProxy atproxy) {
		serviceName_ = serviceName;
		String descURL = "http://123.123.123.34/hgoe.xml";
		if (serviceName == null) {
			System.out.println("No service Name provided.");
		} else {
			/*
			 * Finds...
			 */
			
			
			
			
			/*
			 * Found! Then, the ATUPNPProxy creates an actor for the service
			 * and exports it. 
			 */
			setupProxyActor(serviceName, descURL, atproxy);
		}
	}
	
	public interface ATUPnPProxy {
		public void createAndExportProxyActor(String serviceName, String descriptionURL);
	}

	
	public void setupProxyActor(String serviceName, String descriptionURL, ATUPnPProxy atproxy) {		
		/*
		 * The ATUPnPProxy creates an actor for the service
		 * The actor creates NPnPService instance, that is responsible for 
		 * dealing with SOAP request and response.
		 */
		atproxy.createAndExportProxyActor(serviceName, descriptionURL);
	}

	public interface ATUPnPProxyActor {
		
	}
	
	/*
	 * Configures proxy actor using Java feature.
	 * This may be not used, if all configuration are done in AmbientTalk.
	 */
	public void configureProxyActor(ATUPnPProxyActor atpActor) {
		
		System.out.println("configuring: " + atpActor);
	}
}
