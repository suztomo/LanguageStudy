package net.suztomo.at.upnp;

import java.io.IOException;
import java.net.MalformedURLException;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Iterator;

import net.sbbi.upnp.messages.ActionMessage;
import net.sbbi.upnp.messages.ActionResponse;
import net.sbbi.upnp.messages.UPNPMessageFactory;
import net.sbbi.upnp.messages.UPNPResponseException;
import net.sbbi.upnp.services.*;

/*
 * UPnPService
 * Responsible for
 * 1. retrieving the service's description
 * 2. preparing HashMap<method, URLs>
 * 3. sending SOAP request to the URL 
 */
public class ATUPnPService {
	private HashMap<String, String> methodURLs;/* methodName -> URL */
	private String serviceType_; /* written in device description <serviceType> */
	private UPNPService service_;
	
	/*
	 * 6 arguments to create this instance independently from UPnPProxy.
	 */
	public ATUPnPService(String serviceType, String serviceId, String SCPDURL, String controlURL,
			  String eventSubURL, String USN) {
		init(serviceType, serviceId, SCPDURL, controlURL, eventSubURL, USN);
	}
	
	private void init(String serviceType, String serviceId, String SCPDURL, String controlURL,
			  String eventSubURL, String USN) {
		try {
			service_ = new UPNPService(serviceType, serviceId, SCPDURL, controlURL, eventSubURL, USN);
		} catch (MalformedURLException e) {
			System.out.println(e);
		}
		serviceType_ = serviceType;
	}
	
	public ATUPnPService(String serviceType, String[] otherArgs) {
		init(serviceType, otherArgs[1], otherArgs[2], otherArgs[3], otherArgs[4], otherArgs[5]);
	}
	
	
	private static HashMap<String, String> getMethodURLs(String descriptionURL) {
		HashMap<String, String> m = new HashMap<String, String>();
		return m;
	}
	
	/*
	 * Sends message and wait the result
	 * This uses SOAP protocol (HTTP/TCP)
	 * The return value will be Future in AmbientTalk.
	 */
	@SuppressWarnings("unchecked")
	public String sendMessage(String actionName, Object args) {
		UPNPMessageFactory factory = UPNPMessageFactory.getNewInstance( service_ );
		System.out.println("Sending Message: " + actionName);
		ActionMessage action = factory.getMessage( actionName );
		if (action != null) {
			System.out.println("action found!");
			try {
				ActionResponse resp = action.service();
				System.out.println( "Action returned");
		        System.out.println( resp.getOutActionArgumentValue("NewDefaultConnectionService") );
		        return resp.getOutActionArgumentValue("NewDefaultConnectionService");
			} catch (UPNPResponseException e) {
				/// mmm...
				System.out.println(e);
			} catch (IOException e) {
				System.out.println(e);
			}
		} else {
			System.out.println("No such action " + actionName);
			return "error";
		}
		/*
		try {
			Thread.sleep(4000);
		} catch (InterruptedException e) {
			System.out.println(e);
		}*/
		return "hgoe";
	}
	
	public String getServiceType() {
		return service_.getServiceType();
	}

	@SuppressWarnings("unchecked")
	public String[] getActionNames() {
		String[] ret = new String[service_.getAvailableActionsSize()]; 
		Iterator<String> iter = service_.getAvailableActionsName();
		int c = 0;
		while(iter.hasNext()) {
			ret[c] = iter.next();
		}
		return ret;
	}
	
}
