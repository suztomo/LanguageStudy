package net.suztomo.at.upnp;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import net.sbbi.upnp.Discovery;
import net.sbbi.upnp.devices.UPNPRootDevice;
import net.sbbi.upnp.messages.ActionMessage;
import net.sbbi.upnp.messages.ActionResponse;
import net.sbbi.upnp.messages.UPNPMessageFactory;
import net.sbbi.upnp.messages.UPNPResponseException;
import net.sbbi.upnp.services.UPNPService;



public class ATUPnPProxy {
	private String allServiceKeyword_; 
	
	public ATUPnPProxy(String allServiceKeyword) {
		allServiceKeyword_ = allServiceKeyword;
	}
	
	public void setupProxyActor(Object o) {
		System.out.println("Setting up " + o);
	}
	
	
	
	
	/*
	 * Finds UPnP devices and services.
	 * Setups proxy actor for each service
	 */
	@SuppressWarnings("unchecked")
	public void discover(String serviceName, ATProxy atproxy) {
		if (serviceName == null) {
			System.out.println("No service Name provided.");
		} else {
			/*
			 * Finds...
			 */
			UPNPRootDevice[] devices;
			try {
				devices = Discovery.discover("upnp:rootdevice");
			} catch (IOException e) {
				return;
			}
			if ( devices != null ) {
				for ( int i = 0; i < devices.length; i++ ) {
					UPNPRootDevice d = devices[i];
					List services = d.getServices();
					Iterator iter = services.iterator();
					while(iter.hasNext()) {
						UPNPService service = (UPNPService)iter.next();
						String serviceType = service.getServiceType(); 
						if (!allServiceKeyword_.equals(serviceName) && !serviceType.equals(serviceName)) {
							continue;
						}
						
						String[] serviceInstanceArguments = new String[6];
						serviceInstanceArguments[0] = service.getServiceType();
						serviceInstanceArguments[1] = service.getServiceId();
						serviceInstanceArguments[2] = service.getSCPDURL().toString();
						serviceInstanceArguments[3] = service.getControlURL().toString();
						serviceInstanceArguments[4] = service.getEventSubURL().toString();
						serviceInstanceArguments[5] = service.getUSN().toString();
						if (atproxy != null) 
							setupProxyActor(atproxy, serviceType,serviceInstanceArguments);

					}
					//UPNPService s = new UPNPService(null, , d)
				}
			} else {
				System.out.println("Found no devices..");
			}
			
			
			
			/*
			 * Found! Then, the ATUPNPProxy creates an actor for the service
			 * and exports it. 
			 */
//			setupProxyActor(serviceName, descURL, atproxy);
		}
	}
	
	public interface ATProxy {
		public void createAndExportProxyActor(String serviceType, String[] serviceInstanceArguments);
	}

	
	public void setupProxyActor(ATProxy atproxy, String serviceType, String[] serviceInstanceArguments) {		
		/*
		 * The ATUPnPProxy creates an actor for the service
		 * The actor creates NPnPService instance, that is responsible for 
		 * dealing with SOAP request and response.
		 */
		atproxy.createAndExportProxyActor(serviceType, serviceInstanceArguments);
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
