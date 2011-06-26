package net.suztomo.ponta

import scala.actors._, Actor._
import scala.util.matching._
import scala.actors.Actor.State._

import javax.net.ssl.SSLSocketFactory
import java.net._
import java.io._
import javax.net.ssl._
import javax.security.cert._
import java.security._

import scala.tools.nsc.Interpreter._
import scala.util.Random

/*
 * Creates IrcWriter and reads messages from sockets
 */

case class Stop()

class IrcClient(var config:Options) extends Actor {
  var sock:Socket = null
  var out:PrintWriter = null
  var in:BufferedReader = null
  var nick = config.ircNick
  val ircWriter:IrcWriter = new IrcWriter
  val pinger:IrcPinger = new IrcPinger(ircWriter, config)
  def getPinger():IrcPinger = { pinger }
  def getWriter():IrcWriter = {
    ircWriter
  }
  var rand_generator = new Random(config.ircNick.length)

  def updateSocket() {
    sock = if (! config.ircSsl) {
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
    val ostream = new OutputStreamWriter(sock.getOutputStream(), "UTF-8")
    out = new PrintWriter(ostream, true)
    in  = new BufferedReader(new InputStreamReader(sock.getInputStream(), "UTF-8"))
    if (out == null) {
      println("output is null! : " + sock.toString())
    }
  }
  var reconnectCount:Int = 0

  def act() {
    ircWriter.start
    while(reconnectCount < 100) {
      try {
        log("asking ircConnectionTimer")
        while(! ircConnectionTimer.passed()) {
          log("rejected")
          Thread.sleep(3000 + rand_generator.nextInt(5000))
        }
        log("accepted")
        updateSocket()
        log("updated socket")

        if (ircWriter.getState == Terminated) {
          sock.close()
          exit
        } else {
          var line = in.readLine()
          ircWriter ! UpdateOutput(out, sock)
          while (line != null) {
            serverMsg(line)
            line = in.readLine()
          }
          log("Closed socket normally. It will reconnect the server")
        }
      } catch {
        case e: SocketException => {
          println("connection refused: " + e.toString)
          Thread.sleep(3000)
        }
        case e: SSLHandshakeException => {
          println("SSL handshake failed" + e.toString)
          Thread.sleep(10000)
        }
        case e: IOException => {
          println("ioexception!")
        }
      }
      reconnectCount += 1
      println("Reconnecting...(%d)".format(reconnectCount))
    }
    println("too many fails")
    pinger ! Stop()
  }
  /*
   * This handler should be set before concurrent execution
   */
  var privMsgHandler:(String, String, String) => Unit = {
    (a, b, c) => Unit
  }
  def setMsgHandler(handler:(String, String, String) => Unit) {
    privMsgHandler = handler
  }

  def afterNickAccepted() {
    ircWriter ! Join(config.ircRoom)
    ircWriter ! OutputBufferOk()
    pinger.start
  }

  var ircCommandPattern: Regex = ":(.+? )(.+?) (.+)?".r
  var prefixPattern: Regex = "(.+?)!(.+?)".r
  var ircServerCommandPattern: Regex = "(.+?) (.+)?".r
  var privMsgParamPattern: Regex = "(.+?) :(.+)".r
  var foundYourHostnamePattern: Regex = ".*Found your hostname.*".r
  var connectedSecurelyPattern: Regex = ".*Connected securely.*".r

  def log(str:Object) {
    logger.log(nick + ":" + str.toString)
  }

  def serverMsg(lineRow:String) {
    // :suztomo_!~suztomo@pw126213193203.39.tik.panda-world.ne.jp PRIVMSG #cs2009 :test

    val line = lineRow //new String(lineRow.getBytes(encoding), encoding);
    log(line)
    line match {
      case this.ircCommandPattern(prefix, cmd, params) => {
        cmd match {
          case "PRIVMSG" => {
            params match {
              case this.privMsgParamPattern(channelName, text) => {
                val this.prefixPattern(nick, detailInfo) = prefix
                log("invoking privmsghandler")
                privMsgHandler(nick, channelName, text)
              }
            }
          }
          case "NOTICE" => {


            params match {
              case this.connectedSecurelyPattern() => {
              }
              case this.foundYourHostnamePattern() => {
                ircWriter ! Nick(nick)
                ircWriter ! User(config.ircNick, "0", "*", "http://cs2009irc.appspot.com/")
              }
              case this.privMsgParamPattern(channelName, text) => {
              }
            }
          }
          case "001" => {
            afterNickAccepted()
          }
          case "432" => {
            val r = "[^0-9a-zA-Z{}]".r
            nick = r.replaceAllIn(nick, "-")
            config.ircNick = nick
            ircWriter ! Nick(nick)
          }
          case "433" => { // ERR_ERRONEUSNICKNAME
            nick = nick + '_'
            config.ircNick = nick
            ircWriter ! Nick(nick)
          }
          case _ => {
//            println(">>> Ignoring command(1):" + cmd)
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
//            println(">>> Ignoring command(2):" + line)
          }
        }
      }
      case _ => {
        println(">>> No match line: " + line)
      }
    }
  }
}


class IrcPinger(val ircWriter:IrcWriter, val config:Options) extends Actor {
  def act() {
    loop {
      reactWithin(30 * 1000) {
        case TIMEOUT => {
          ircWriter ! Nick(config.ircNick)
          Thread.sleep(30 * 1000)
        }
        case Stop() => {
          exit
        }
      }
    }
  }
}
