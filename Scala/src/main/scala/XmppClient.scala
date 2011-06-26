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
  val chatManager = connection.getChatManager()
  val chats:Array[Chat] = new Array(config.xmppRecipients.length)
  var chat:Chat = null
  var count:Int = 0
  for (recipient <- config.xmppRecipients) {
    chat = chatManager.createChat(recipient, null)
    chats(count) = chat
    count += 1
  }

  val xmppWriter = new XmppWriter(connection, chats)

  /*
   * This handler should be set before concurrent execution
   */
  var msgHandler:(String, String, XmppClient) => Unit = null
  def setMsgHandler(handler:(String, String, XmppClient) => Unit) {
    msgHandler = handler
  }
//  val filter:PacketFilter = new AndFilter(new PacketTypeFilter(classOf[Message]), 
//        new FromContainsFilter("tomotomotomo888@gmail.com"));
  val filter:PacketFilter = new PacketTypeFilter(classOf[Message])
  // Assume we've created an XMPPConnection name "connection".

  def getWriter():XmppWriter = {
    xmppWriter
  }

  def act() {
    xmppWriter.start
    val collector = connection.createPacketCollector(filter);
    while (true) {
      val packet: Packet = collector.nextResult();
      if(packet.isInstanceOf[Message]) {
        val msg: Message = packet.asInstanceOf[Message]
        val body_sender =  msg.getBody().split(":", 2)
        val body = body_sender(1)
        val sender = body_sender(0)
        if (msgHandler != null) {
          msgHandler(sender, body, this)
        }
      }
    }

  }
}
