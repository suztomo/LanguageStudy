package net.suztomo.at.upnp;

import java.io.IOException;
import java.net.MalformedURLException;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

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
	static private Log log = LogFactory.getLog(ATUPnPService.class);
	
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

	public interface ATUPnPEnvelope {
		public Object getValue(String key);
		public Object[] getTable();
		public void clear();
		public void setValue(String key, Object value);
		public void setError(String message);
	}
	
	private static Boolean prepareActionEnvelope(ActionMessage action, ATUPnPEnvelope envelope) {
		List params = action.getInputParameterNames();
		if (params == null) {
			log.debug("no arguments for " + action.toString());
			return true;
		}
		Iterator iter = params.iterator();
		while(iter.hasNext()) {
			String key = (String) iter.next();
			Object val = envelope.getValue(key);
			if (val == null) {
				prepareEnvelopeWithError(envelope, "Invalid arguments for " + action.toString());				
				return false;
			}
			String className = val.getClass().getName();
			if (className.equals("java.lang.String")) {
				action.setInputParameter(key, (String)val);
			} else if (className.equals("java.lang.Integer")){
				action.setInputParameter(key, (Integer)val);
			} else if (className.equals("java.lang.Double")) {
				action.setInputParameter(key, (Double)val);
			} else if (className.equals("java.lang.Boolean")){
				action.setInputParameter(key, (Boolean)val);
			} else {
				prepareEnvelopeWithError(envelope, "Unsupported arguments " + key + " for " + action.toString());				
				return false;
			}
		}
		return true;
	}
	private static Boolean prepareResponseEnvelope(ActionResponse response, ATUPnPEnvelope envelope) {
		Set params = response.getOutActionArgumentNames();
		Iterator iter = params.iterator();
		envelope.clear();

		while(iter.hasNext()) {
			String key = (String) iter.next();
			Object value = response.getOutActionArgumentValue(key);
			envelope.setValue(key, value);
		}
		return true;
	}
	
	private static void prepareEnvelopeWithError(ATUPnPEnvelope envelope, String message) {
		envelope.clear();
		envelope.setError(message);
	}
	
	/*
	 * Sends message and wait the result
	 * This uses SOAP protocol (HTTP/TCP)
	 * The return value will be Future in AmbientTalk.
	 */
	@SuppressWarnings("unchecked")
	public ATUPnPEnvelope sendMessage(String actionName, ATUPnPEnvelope envelope) {
		UPNPMessageFactory factory = UPNPMessageFactory.getNewInstance( service_ );
		ActionMessage action = factory.getMessage( actionName );
		if (action != null) {
			Boolean r = prepareActionEnvelope(action, envelope);
			if (!r) {
				/* argument error */
				return envelope;
			}
			try {
				ActionResponse resp = action.service();				
		        prepareResponseEnvelope(resp, envelope);
			} catch(UPNPResponseException e) {
				System.out.println(e);
				envelope.setError(e.toString());
			} catch  (IOException e) {
				envelope.setError(e.toString());
			}
		} else {
			prepareEnvelopeWithError(envelope, "No such action");
		}
		/*
		try {
			Thread.sleep(4000);
		} catch (InterruptedException e) {
			System.out.println(e);
		}*/
		return envelope;
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
