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
        return r;
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

    function moveSVGRect(node, x, y) {
        node.setAttribute('x', x);
        node.setAttribute('y', y);
    }

    function createActor(vm, id) {
        var instance = {
            vm : vm,
            x : 55,
            y : 55,
            aid : id,
            svgNode : undefined,
            color: "#BB1111",
            init : function() {
                if (vm == undefined) {
                    console.log("Actor not spedified by Virtual Machine");
                }
                if (id == undefined) {
                    console.log("Actor not spedified by Actor id");
                }
                svgNode = createCircle({
                    x: 50, y: 50, r: 20, fill: this.color
                });
                ActorCanvas.appendChild(svgNode);
            },
            draw: function () {
                moveSVGCircle(svgNode, this.x, this.y);
            },
            update: function () {
                
            }
        };
        instance.init();
        return instance;
    }

    function createVM(id) {
        var instance = {
            x : 55,
            y : 55,
            vmid: id,
            svgNode : 0,
            color: "#CCC",
            dragStatus : 0,
            dragRelativeX : 0,
            dragRelativeY : 0,
            mouseX: 0,
            mouseY: 0,
            init : function(self) {
                if (id == undefined) {
                    console.log("Virtual Machine not spedified by VMID");
                }
                var svgNode = createRoundedRect({
                    x: 0, y: 0, rx: 5, ry: 5, fill: this.color,
                    height: 150, width: 220
                });
                svgNode.setAttribute('draggable', true);
                svgNode.vm = self;
                VMCanvas.appendChild(svgNode);
                svgNode.addEventListener("mousedown", this.onMouseDown, true);
                svgNode.addEventListener("mousemove", this.onMouseMove, true);
                svgNode.addEventListener("mouseup", this.onMouseUp, true);
                self.svgNode = svgNode;
            },
            draw: function () {
                moveSVGRect(this.svgNode, this.x, this.y);
            },
            update: function () {
                if (this.dragStatus == 0) {
                    
                } else if (this.dragStatus == 1) {
                    this.x = this.mouseX - this.dragRelativeX;
                    this.y = this.mouseY - this.dragRelativeY;
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
            },
            onMouseMove: function(evt) {
                self = evt.target.vm;
                self.mouseX = evt.offsetX;
                self.mouseY = evt.offsetY;
            },
            onMouseUp: function(evt) {
                self = evt.target.vm;
                self.dragStatus = 0;
            }
        };
        instance.init(instance);
        return instance;
    }
    
    ca = createActor;
    cv = createVM;


    ActorManager = {
        actors : [],
        draw : function () {
            for (var i=0; i<actors.length; ++i) {
                var a = actors[i];
                a.draw();
            }
        },
        update : function () {
            for (var i=0; i<actors.length; ++i) {
                var a = actors[i];
                a.update();
            }
        }
    };

    VMManager = {
        vms : [],
        draw: function(self) {
            for (var i=0; i< self.vms.length; ++i) {
                var v = self.vms[i];
                v.draw(v);
            }
        },
        update : function(self) {
            for (var i=0; i< self.vms.length; ++i) {
                var v = self.vms[i];
                v.update(v);
            }
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
        VMManager.update(VMManager);
        VMManager.draw(VMManager);
        /*
        for (var i=0; i<circles.length; i++) {
            var c = circles[i];
            moveCircle(c);
        }*/
    };
    var frame = 10;
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


    var v = createVM("tako");
    VMManager.vms = [v];
    repeator();
});