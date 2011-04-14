package net.suztomo.ponta

import scala.collection.JavaConversions._
import org.jivesoftware.smack._
import scala.actors._, Actor._

/*
 * sends messages to all chats
 */
case class XmppMsg(text:String)
case class NewConnection(newConnection:XMPPConnection)
class XmppWriter(var connection:XMPPConnection,
                 val chats:Array[Chat]) extends Actor {
  def act() {
    loop{
      react{
        case NewConnection(newConnection:XMPPConnection) => {
          connection = newConnection
        }
        case Msg(text) => {
          for(chat <- chats) {
            chat.sendMessage(text)
          }
        }
      }
    }
  }
}
