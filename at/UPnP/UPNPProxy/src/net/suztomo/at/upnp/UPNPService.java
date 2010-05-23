package net.suztomo.at.upnp;

import java.util.HashMap;
import java.util.ArrayList;

public class UPNPService {
	private HashMap<String, String> methodURLs;/* methodName -> URL */
	private String descriptionURL_;
	
	public UPNPService(String serviceName, String descriptionURL) {
		descriptionURL_ = descriptionURL;
	}
	
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
	public void sendMessage(String methodName, ArrayList args) {
		
	}
	
}
