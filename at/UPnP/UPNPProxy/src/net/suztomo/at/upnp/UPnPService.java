package net.suztomo.at.upnp;

import java.util.HashMap;
import java.util.ArrayList;


/*
 * UPnPService
 * Responsible for
 * 1. retrieving the service's description
 * 2. preparing HashMap<method, URLs>
 * 3. sending SOAP request to the URL 
 */
public class UPnPService {
	private HashMap<String, String> methodURLs;/* methodName -> URL */
	private String serviceType_; /* written in device description <serviceType> */
	private String descriptionURL_; /* written in device description <SCPDURL> */
	
	public UPnPService(String serviceType, String descriptionURL) {
		serviceType_ = serviceType;
		descriptionURL_ = descriptionURL;
	}
	
	/*
	 * Prepare methodURLs, that maintain method name and 
	 * 
	 */
	public void init() {
		methodURLs = getMethodURLs(descriptionURL_);
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
	public int sendMessage(String actionName, Object args) {
		try {
			Thread.sleep(4000);
		} catch (InterruptedException e) {
			System.out.println(e);
		}
		return 100;
	}
	
}
