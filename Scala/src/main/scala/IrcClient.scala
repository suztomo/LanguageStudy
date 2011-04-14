package net.suztomo.ponta

import scala.actors._, Actor._
import scala.util.matching._

import javax.net.ssl.SSLSocketFactory
import java.net.Socket
import java.io._
import javax.net.ssl._
import javax.security.cert._
import java.security._

import scala.tools.nsc.Interpreter._

/*
 * Creates IrcWriter and reads messages from sockets
 */
class IrcClient(var config:Options) extends Actor {
  val sock = if (! config.ircSsl) {
    new Socket(config.ircServer, config.ircPort)
  } else {
    val fc = SSLSocketFactory.getDefault
/*
      if (config.ignoreCertificate) {

      var km:Array[KeyManager] = Array();
      val tm:Array[TrustManager] = Array(
        new MyX509TrustManager()
      )
      val sslcontext= SSLContext.getInstance("SSL");
      sslcontext.init(km, tm, new SecureRandom());
      sslcontext.getSocketFactory()
    } else {
      
    }
    * */
    fc.createSocket(config.ircServer, config.ircPort).asInstanceOf[
      javax.net.ssl.SSLSocket]
  }
  val out = new PrintWriter(sock.getOutputStream(), true)
  val in = new BufferedReader(new InputStreamReader(sock.getInputStream(), "UTF-8"))

  val ircWriter = new IrcWriter(out)

  def act() {
    ircWriter.start
    ircWriter ! User("suztomobot", "test", "test", "suztomobot")
    ircWriter ! Nick("suztomobot")
    ircWriter ! Join("#cs2009")

    var line = in.readLine()
    while (line != null) {
      serverMsg(line)
      line = in.readLine()
    }
  }

  /*
   * This handler should be set before concurrent execution
   */
  var privMsgHandler:(String, String, String) => Unit = null
  def setMsgHandler(handler:(String, String, String) => Unit) {
    privMsgHandler = handler
  }

  var ircCommandPattern: Regex = ":(.+? )(.+?) (.+)?".r
  var prefixPattern: Regex = "(.+?)!(.+?)".r
  var ircServerCommandPattern: Regex = "(.+?) (.+)?".r
  var privMsgParamPattern: Regex = "(.+?) :(.+)".r
  def serverMsg(lineRow:String) {
    // :suztomo_!~suztomo@pw126213193203.39.tik.panda-world.ne.jp PRIVMSG #cs2009 :test

    val line = lineRow //new String(lineRow.getBytes(encoding), encoding);
    line match {
      case this.ircCommandPattern(prefix, cmd, params) => {
        cmd match {
          case "PRIVMSG" => {
            params match {
              case this.privMsgParamPattern(channelName, text) => {
                val this.prefixPattern(nick, detailInfo) = prefix
                privMsgHandler(nick, channelName, text)
              }
            }
          }
          case "NOTICE" => {
            params match {
              case this.privMsgParamPattern(channelName, text) => {
                println("NOTICE! :" + text)
              }
            }
          }
          case _ => {
            println("Ignoring command(1):" + cmd)
          }
        }
      }
      case this.ircServerCommandPattern(cmd, params) => {
        cmd match {
          case "PING" => {
            /* Ping should be responded by Pong
             * e.g. PING :hitchcock.freenode.net */
            ircWriter ! Msg("PONG " + line.subSequence(5, line.size))
          }
          case _ => {
            println("Ignoring command(2):" + line)
          }
        }
      }
      case _ => {
        println("No match line: " + line)
      }
    }
  }
}
 
