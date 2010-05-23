package net.suztomo.at.upnp;
import java.util.HashMap;


public class UPNPProxy {
	private String serviceName_;
	private HashMap<String, UPNPService> services; /* String -> UPnPService */
	
	public UPNPProxy() {
		
	}
	
	public void setupProxyActor(Object o) {
		System.out.println(o);
	}
	
	public interface ATUPNPProxy {
		public void setupService(String serviceName);
	}
	
	public void discover(String serviceName, ATUPNPProxy o) {
		serviceName_ = serviceName;
		o.setupService(serviceName_);
	}
}
