jQuery(function ($) {
	// redirect the old bitches
    var SVGRoot;

    function resizeSVGRoot() {
        if (SVGRoot == null) {
            return alert('SVGRoot is null');
        }
        SVGRoot.setAttribute('width', "100%");
        SVGRoot.setAttribute('height', "100%");
    }

    function init() {
        targetCircle = document.getElementById('targetCircle');
        SVGRoot = document.getElementById('SVGRoot');
        resizeSVGRoot();
    }

    function onCircleClick(obj) {
        var target = obj.currentTarget;
        moveCircle(obj);
        repeat = false;
    }

    function moveCircle(svgCircle) {
        svgCircle.setAttribute("cx", svgCircle.getAttribute("cx") - 1);
    }


    var circles = []
    var el = document.getElementById('processing');
    var p = Processing(el);

    setup=function() {
        var cs = document.getElementsByClassName('circle');
        for (var i=0; i<cs.length; i++) {
            circles[i] = cs[i];
        }
    };
    draw=function() {
        for (var i=0; i<circles.length; i++) {
            var c = circles[i];
            moveCircle(c);
        }
    };
    var frame = 100;
    var repeat = true;
    setup();
    function repeator() {
        draw();
        if (repeat)
            setTimeout(repeator, frame);
    }
    $('#targetCircle').click(onCircleClick);
    repeator();
});