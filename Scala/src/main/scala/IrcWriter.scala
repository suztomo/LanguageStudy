package net.suztomo.ponta

import scala.actors._, Actor._
import java.io._


case class Msg(line:String)
case class PrivMsg(line:String, text:String)
case class Join(channel : String)
case class Nick(nickname:String)
case class User(username:String, hostname:String, servername:String, realname:String)

/*
 * Sends message to IRC connection
 * from MrToms
 */
class IrcWriter(val out:PrintWriter) extends Actor {
  var xmppWriter:XmppWriter = null
  def setXmppClient(xmppWriter:XmppWriter) {
    this.xmppWriter = xmppWriter
  }
  def write(line:String) {
    out.println(line)
    println(">>> " + line)
  }

  def act() {
    loop {
      react {
        case Msg(line) => {
          /* general messages */

        }
        case PrivMsg(channel, text) => {
          write("PRIVMSG " + channel + " :" + text)
        }
        case Join(channel:String) => {
          write("JOIN " + channel)
        }
        case User(username, hostname, servername, realname) => {
          write("USER %s %s %s %s".format(username, hostname,
                                                servername, realname))
        }
        case Nick(nickname) => {
          write("NICK " + nickname)
        }
      }
    }
  }
}
