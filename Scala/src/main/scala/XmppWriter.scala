package net.suztomo.ponta

import scala.collection.JavaConversions._
import org.jivesoftware.smack._
import scala.actors._, Actor._

/*
 * sends messages to all chats
 */
case class XmppBroadcastMsg(text:String)
case class XmppBroadcastMsgFrom(text:String, sender:String)
case class XmppMsg(chat:Chat, text:String)
case class NewConnection(newConnection:XMPPConnection)
class XmppWriter(var connection:XMPPConnection,
                 val chats:Array[Chat]) extends Actor {
  def chatManager:ChatManager = connection.getChatManager()
  var chat:Chat = null
  def act() {
    loop{
      react{
        case NewConnection(newConnection:XMPPConnection) => {
          connection = newConnection
        }
        case XmppBroadcastMsg(text) => {
          for(chat <- chats) {
            chat.sendMessage(text)
          }
        }
        case XmppMsg(chat, text) => {
          chat.sendMessage(text)
        }
      }
    }
  }
}
