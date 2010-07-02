jQuery(function ($) {
	// redirect the old bitches
    var SVGRoot;
    console.log(2);

    function checkAttribute(attr, required, message) {
        for (var i=0; i<required.length; i++) {
            if (attr[required[i]] == undefined) {
                console.log(message);
                console.log(attr);
            }
        }
    }

    function createCircle(attr) {
        var c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        checkAttribute(attr, ['x', 'y', 'r', 'fill'], "invalid attribute for circle");
        c.setAttribute('cx', attr['x']);
        c.setAttribute('cy', attr['y']);
        c.setAttribute('r', attr['r']);
        c.setAttribute('fill', attr['fill']);
        return c;
    }


    function createRoundedRect(attr) {
        var r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        var required = ['x', 'y', 'width', 'height', 'rx', 'ry', 'fill'];
        checkAttribute(attr, required, "invalid attribute for rounded rect");
        r.setAttribute('x', attr['x']);
        r.setAttribute('y', attr['y']);
        r.setAttribute('width', attr['width']);
        r.setAttribute('height', attr['height']);
        r.setAttribute('rx', attr['rx']);
        r.setAttribute('ry', attr['ry']);
        r.setAttribute('fill', attr['fill']);
        r.setAttribute('stroke', '#333');
        r.setAttribute('stroke-width', 1);
        return r;
    }

    function createText(attr) {
        checkAttribute(attr, ["text", "x", "y", "size"], "invalid attribute for text");
        var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t.setAttribute('x', attr['x']);
        t.setAttribute('y', attr['y']);
        t.setAttribute('style', "font-family:Verdana;font-size:" +  attr['size']);
        t.appendChild(document.createTextNode(attr['text']));
        return t;
    }

    function init() {
        targetCircle = document.getElementById('targetCircle');
        SVGRoot = document.getElementById('SVGRoot');
        ActorCanvas = document.getElementById('ActorCanvas');
        VMCanvas = document.getElementById('VMCanvas');
        PortCanvas = document.getElementById('PortCanvas');
        canvases = [ActorCanvas, VMCanvas, PortCanvas];
        setCanvasPositions();
        e = createCircle({
            x: 100, y:120, r: 30, fill: '#99AA88'
        });
        SVGRoot.appendChild(e);
        r = createRoundedRect({
            x: 120, y: 80, rx: 5, ry: 5, fill: "#BBFF77",
            height: 50, width: 30
        });
        SVGRoot.appendChild(r);
    }

    function setCanvasPositions() {
        for (var i=0; i<canvases.length; ++i) {
            var c = canvases[i];
            c.setAttribute('width', '100%');
            c.setAttribute('height', '100%');
        }
    }

    function moveSVGCircle(node, x, y) {
        node.setAttribute('cx', x);
        node.setAttribute('cy', y);
    }

    function moveSVG(node, x, y) {
        node.setAttribute('x', x);
        node.setAttribute('y', y);
    }

    function createActor(vm, id) {
        var instance = {
            vm : vm,
            x : 0,
            y : 0,
            vx : 1,
            vy : 1,
            ax : 0,
            ay : 0,
            fx : 0,
            fy : 0,
            m : 5,
            containerX : 0,
            containerY : 0,
            containerWidth: 0,
            containerHeight: 0,
            aid : id,
            svgNode : null,
            containerInner: 0.2,
            color: "#BB1111",
            init : function() {
                if (vm == undefined) {
                    console.log("Actor not spedified by Virtual Machine");
                }
                if (id == undefined) {
                    console.log("Actor not spedified by Actor id");
                }
                this.svgNode = createCircle({
                    x: 50, y: 50, r: 20, fill: this.color
                });
                this.containerWidth = Math.floor(vm.width * (1- 2 * this.containerInner));
                this.containerHeight = Math.floor(vm.height *  (1- 2 * this.containerInner));
                ActorCanvas.appendChild(this.svgNode);
                this.x = Math.floor(Math.random() * 100);
                this.y = Math.floor(Math.random() * 100);
            },
            draw: function () {
                moveSVGCircle(this.svgNode, this.x, this.y);
            },
            checkBoundary: function() {
                this.containerX = vm.x + this.containerWidth * this.containerInner;
                this.containerY = vm.y + this.containerHeight * this.containerInner;
                if (this.x < this.containerX) {
                    this.fx += (this.containerX - this.x) / 5;
                }
                if (this.x > this.containerWidth + this.containerX) {
                    this.fx += - (this.x
                                 - (this.containerWidth + this.containerX) ) / 5;
                }
                if (this.y < this.containerY) {
                    this.fy += (this.containerY - this.y) / 5;
                }
                if (this.y > this.containerHeight + this.containerY) {
                    this.fy += - (this.y
                                 - (this.containerHeight + this.containerY)) / 5;
                }
            },
            update: function () {
                this.checkBoundary();
                this.ax = this.fx / this.m;
                this.ay = this.fy / this.m;
                this.vx += this.ax;
                this.vy += this.ay;
                this.vx -= this.vx / 10;
                this.vy -= this.vy / 10;
                this.x += Math.floor(this.vx);
                this.y += Math.floor(this.vy);

                this.fx = 0;
                this.fy = 0;
            }
        };
        instance.init();
        return instance;
    }

    function createVM(id) {
        var instance = {
            x : 0,
            y : 0,
            width: 280,
            height: 220,
            vx : 1,
            vy : 1,
            ax : 0,
            ay : 0,
            fx : 0,
            fy : 0,
            m : 10,
            containerX : 0,
            containerY : 0,
            containerWidth: 0,
            containerHeight: 0,
            vmid: id,
            svgNode : 0,
            svgTextNode: 0,
            color: "#CCC",
            dragStatus : 0,
            dragRelativeX : 0,
            dragRelativeY : 0,
            mouseX: 0,
            mouseY: 0,
            actorManager: null,
            fontSize: 36,
            init : function(self) {
                if (id == undefined) {
                    console.log("Virtual Machine not spedified by VMID");
                }
                var svgNode = createRoundedRect({
                    x: 0, y: 0, rx: 5, ry: 5, fill: this.color,
                    height: this.height, width: this.width
                });
                svgNode.setAttribute('draggable', true);
                svgNode.vm = self;
                VMCanvas.appendChild(svgNode);
                svgNode.addEventListener("mousedown", this.onMouseDown, true);
                svgNode.addEventListener("mousemove", this.onMouseMove, true);
                svgNode.addEventListener("mouseup", this.onMouseUp, true);
                svgNode.setAttribute('title', "" + self.vmid);
                self.svgNode = svgNode;
                self.x = Math.floor(Math.random() * 300);
                self.y = Math.floor(Math.random() * 200);
                self.textNode
                var t = createText({x: self.x, y: self.y, text: self.vmid,
                                    size: this.fontSize});
                VMCanvas.appendChild(t);
                self.svgTextNode = t;
                self.actorManager = createActorManager();
            },
            draw: function () {
                moveSVG(this.svgNode, this.x, this.y);
                moveSVG(this.svgTextNode, this.x + 10, this.y+ 5 + this.fontSize);
                this.actorManager.draw();
            },
            update: function () {
                this.actorManager.update();
                if (this.dragStatus == 0) {
                    this.checkBoundary();
                    this.ax = this.fx / this.m;
                    this.ay = this.fy / this.m;
                    this.vx += this.ax;
                    this.vy += this.ay;
                    this.vx -= this.vx / 10;
                    this.vy -= this.vy / 10;
                    this.x += Math.floor(this.vx);
                    this.y += Math.floor(this.vy);
                } else if (this.dragStatus == 1) {
                    this.x = this.mouseX - this.dragRelativeX;
                    this.y = this.mouseY - this.dragRelativeY;
                }
                this.fx = 0;
                this.fy = 0;
            },
            checkBoundary: function() {
                this.containerX = this.svgNode.parentElement.x.animVal.value;
                this.containerY = this.svgNode.parentElement.y.animVal.value;
                this.containerWidth = this.svgNode.parentElement.width.animVal.value;
                this.containerHeight = this.svgNode.parentElement.height.animVal.value;
                if (this.x < this.containerX) {
                    this.fx += (this.containerX - this.x) / 5;
                }
                if (this.x + this.width > this.containerWidth + this.containerX) {
                    this.fx += - ((this.x + this.width)
                                 - (this.containerWidth + this.containerX) ) / 5;
                }
                if (this.y < this.containerY) {
                    this.fy += (this.containerY - this.y) / 5;
                }
                if (this.y + this.height > this.containerHeight + this.containerY) {
                    this.fy += - ((this.y + this.height)
                                 - (this.containerHeight + this.containerY)) / 5;
                }
            },
            onMouseDown: function(evt) {
                // left click
                if( evt.which != 1 )
                    return;
                self = evt.target.vm;
                self.dragRelativeX = evt.offsetX - self.x;
                self.dragRelativeY = evt.offsetY - self.y;
                ev = evt;
                self.dragStatus = 1;
                e = self
                e.svgNode.parentElement.appendChild(e.svgNode);
                e.svgNode.parentElement.appendChild(e.svgTextNode);
            },
            onMouseMove: function(evt) {
                self = evt.target.vm;
                self.mouseX = evt.offsetX;
                self.mouseY = evt.offsetY;
            },
            onMouseUp: function(evt) {
                self = evt.target.vm;
                self.dragStatus = 0;
            },
            putActor: function(id) {
                var a = createActor(this, id);
                this.actorManager.put(a);
            }
        };
        instance.init(instance);
        return instance;
    }
    
    ca = createActor;
    cv = createVM;

    function createActorManager() {
        return {
            actors : [],
            addressBook : {},
            k : 20,
            springLength: 150,
            draw : function () {
                for (var i=0; i<this.actors.length; ++i) {
                    var a = this.actors[i];
                    a.draw();
                }
            },
            update : function () {
                for (var i=0; i<this.actors.length; ++i) {
                    var a = this.actors[i];
                    for (var j=0; j<this.actors.length; ++j) {
                        if (i == j) continue;
                        var b = this.actors[j];
                        var dx = (b.x - a.x);
                        var dy = (b.y - a.y);
                        var d = Math.sqrt(dx * dx + dy * dy);
                        var f = (d - this.springLength) / this.k;
                        var fx = f * dx / d;
                        var fy = f * dy / d;
                        a.fx += fx;
                        a.fy += fy;
                    }
                    a.update();
                }
            },
            put : function(actor) {
                
                this.actors.push(actor);
                this.addressBook[actor.aid] = actor;
            },
            get : function(aid) {
                return this.addressBook[aid];
            }
        };
    }

    VMManager = {
        vms : [],
        addressBook : {},
        springLength : 500,
        k : 5,
        draw: function() {
            for (var i=0; i< this.vms.length; ++i) {
                var v = this.vms[i];
                v.draw(v);
            }
        },
        update : function() {
            for (var i=0; i< this.vms.length; ++i) {
                var v = this.vms[i];
                for (var j=0; j < this.vms.length; ++j) {
                    if (j == i) continue;
                    var w = this.vms[j];
                    var dx = (w.x - v.x);
                    var dy = (w.y - v.y);
                    var d = Math.sqrt(dx * dx + dy * dy);
                    var f = (d - this.springLength) / this.k;
                    var fx = f * dx / d;
                    var fy = f * dy / d;
                    v.fx += fx;
                    v.fy += fy;
                }
                v.update(v);
            }
        },
        put : function(vm) {
            this.vms.push(vm);
            this.addressBook[vm.vmid] = vm;
        },
        get : function(vmid) {
            return this.addressBook[vmid];
        }
    };

    function onCircleClick(obj) {
        var target = obj.currentTarget;
        moveCircle(obj);
        repeat = false;
    }

    function moveCircle(svgCircle) {
        svgCircle.setAttribute("cx", svgCircle.getAttribute("cx") - 1);
    }


    var circles = []
    /*
    var el = document.getElementById('processing');
    var p = Processing(el);
    */
    setup=function() {
        var cs = document.getElementsByClassName('circle');
        for (var i=0; i<cs.length; i++) {
            circles[i] = cs[i];
        }
    };
    draw=function() {
        VMManager.update();
        VMManager.draw();
        /*
        for (var i=0; i<circles.length; i++) {
            var c = circles[i];
            moveCircle(c);
        }*/
    };
    var frame = 50;
    var repeat = true;
    setup();
    var count = 0;
    function repeator() {
        draw();
        if (repeat)
            setTimeout(repeator, frame);
    }
    $('#targetCircle').click(onCircleClick);
    init();


    v = createVM("tako");
    var v2 = createVM("hoge");
    VMManager.vms = [v, v2];
    for (var i = 0; i< 10; i++) {
        VMManager.put(createVM("vm" + i));
    }
    v.putActor("123");

    v.putActor("suzuki");
    v.putActor("jorge");
    v.putActor("coen");
    v.putActor("kevin");
    v.putActor("elisa");

    repeator();
});