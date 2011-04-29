package net.suztomo.ponta

import scala.actors._, Actor._
import java.io._

case class UpdateOutput(output:PrintWriter)
case class Msg(line:String)
case class PrivMsg(line:String, text:String)
case class Notice(line:String, text:String)
case class Join(channel : String)
case class Nick(nickname:String)
case class User(username:String, hostname:String, servername:String, realname:String)

/*
 * Sends message to IRC connection
 * from MrToms
 */
class IrcWriter() extends Actor {
  var out:PrintWriter = null
  def write(line:String) {
    if (out == null) {
      println("output is not initialized")
      return
    }
    out.println(line)
    println("<<< " + line)
  }

  def act() {
    loop {
      react {
        case UpdateOutput(output) => {
          out = output
        }
        case Msg(line) => {
          /* general messages */
          write(line)
        }
        case PrivMsg(channel, text) => {
          write("PRIVMSG " + channel + " :" + text)
        }
        case Notice(channel, text) => {
          write("NOTICE " + channel + " :" + text)
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
