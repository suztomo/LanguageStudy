package net.suztomo.ponta
import scala.actors._, Actor._
import scala.collection.JavaConversions._

import org.jivesoftware.smack._
import org.jivesoftware.smack.filter._
import org.jivesoftware.smack.packet._

import java.net._

/*
 * Creates XmppWriter and reads messages from XMPP server
 */
class XmppClient(var config:Options) extends Actor {
  val user = config.xmppBot
  val pass = config.xmppPass
  val xmppConfig = new ConnectionConfiguration(config.xmppServer, config.xmppPort,
                                               "gmail.com")
  val connection = new XMPPConnection(xmppConfig)
  try {
    connection.connect()
    println("Connected to " + connection.getHost())
  } catch {
    case e:UnknownHostException => {
      println("Unknown Host:" + config.xmppServer)
    }
    case e:XMPPException => {
      println("Failed to connect to the server")
    }
  }
  try {
	connection.login(user, pass)
	println("Logged in as " + connection.getUser())
  } catch {
    case e:Any => {
	  println("Failed to log in as " + connection.getUser())
    }
  }
  
  val chats:Array[Chat] = new Array(config.xmppRecipients.length)
  var chat:Chat = null
  var count:Int = 0
  for (recipient <- config.xmppRecipients) {
    chat = connection.getChatManager().createChat(recipient, null)
    chats(count) = chat
    count += 1
  }

  val xmppWriter = new XmppWriter(connection, chats)

  /*
   * This handler should be set before concurrent execution
   */
  var msgHandler:(String, String) => Unit = null
  def setMsgHandler(handler:(String, String) => Unit) {
    msgHandler = handler
  }
  val filter:PacketFilter = new AndFilter(new PacketTypeFilter(classOf[Message]), 
        new FromContainsFilter("tomotomotomo888@gmail.com"));
  // Assume we've created an XMPPConnection name "connection".

  // First, register a packet collector using the filter we created.
  val myCollector:PacketCollector = connection.createPacketCollector(filter);
  // Normally, you'd do something with the collector, like wait for new packets.

  // Next, create a packet listener. We use an anonymous inner class for brevity.
  val myListener:PacketListener = new PacketListener() {
    def processPacket(packet:Packet):Unit = {
      // Do something with the incoming packet here.
      val props = packet.getPropertyNames()
      for (i <- props) {
        println("Packet." + i)
      }
    }
  };
  def getWriter():XmppWriter = {
    xmppWriter
  }
  def act() {
    xmppWriter.start
    connection.addPacketListener(myListener, filter);
  }
}
