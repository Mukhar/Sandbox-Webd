	var canvas = document.getElementsByTagName('canvas')[0];
	canvas.width = 900; canvas.height = 900;
	var gkhead = new Image;
	window.onload = function(){		
		var context = canvas.getContext('2d');
		trackTransforms(context);
		function redraw(){
			// Clear the entire canvas
			console.log('redraw');
			var p1 = context.transformedPoint(0,0);
			var p2 = context.transformedPoint(canvas.width,canvas.height);
			context.clearRect(p1.x,p1.y,p2.x-p1.x,p2.y-p1.y);
			context.drawImage(gkhead,0,0);
			
		}
		redraw();
		
		var lastX=canvas.width/2, lastY=canvas.height/2;
		var dragStart,dragged;
		canvas.addEventListener('mousedown',function(evt){
			console.log('mousedown');
			document.body.style.mozUserSelect = document.body.style.webkitUserSelect = document.body.style.userSelect = 'none';
			lastX = evt.offsetX || (evt.pageX - canvas.offsetLeft);
			lastY = evt.offsetY || (evt.pageY - canvas.offsetTop);
			dragStart = context.transformedPoint(lastX,lastY);
			dragged = false;
		},false);
		canvas.addEventListener('mousemove',function(evt){
			console.log('mousemove');
			lastX = evt.offsetX || (evt.pageX - canvas.offsetLeft);
			lastY = evt.offsetY || (evt.pageY - canvas.offsetTop);
			dragged = true;
			if (dragStart){
				var pt = context.transformedPoint(lastX,lastY);
				context.translate(pt.x-dragStart.x,pt.y-dragStart.y);
				redraw();
			}
		},false);
		canvas.addEventListener('mouseup',function(evt){
			console.log('mouseup');
			dragStart = null;
			if (!dragged) zoom(evt.shiftKey ? -1 : 1 );
		},false);

		var scaleFactor = 1.1;
		// zoom on click 
		var zoom = function(clicks){
			var pt = context.transformedPoint(lastX,lastY);
			context.translate(pt.x,pt.y);
			var factor = Math.pow(scaleFactor,clicks);
			context.scale(factor,factor);
			context.translate(-pt.x,-pt.y);
			redraw();
		}

		var handleScroll = function(evt){
			var delta = evt.wheelDelta ? evt.wheelDelta/40 : evt.detail ? -evt.detail : 0;
			if (delta) zoom(delta);
			return evt.preventDefault() && false;
		};
		canvas.addEventListener('DOMMouseScroll',handleScroll,false);
		canvas.addEventListener('mousewheel',handleScroll,false);
	};
	gkhead.src = 'Cy5_001.png';
	
	// Adds context.getTransform() - returns an SVGMatrix
	// Adds context.transformedPoint(x,y) - returns an SVGPoint
	function trackTransforms(context){
		var svg = document.createElementNS("http://www.w3.org/2000/svg",'svg');
		var xform = svg.createSVGMatrix();
		context.getTransform = function(){ return xform; };
		
		var savedTransforms = [];
		var save = context.save;
		context.save = function(){
			savedTransforms.push(xform.translate(0,0));
			return save.call(context);
		};
		var restore = context.restore;
		context.restore = function(){
			xform = savedTransforms.pop();
			return restore.call(context);
		};

		var scale = context.scale;
		context.scale = function(sx,sy){
			xform = xform.scaleNonUniform(sx,sy);
			return scale.call(context,sx,sy);
		};
		// var rotate = context.rotate;
		// context.rotate = function(radians){
		// 	xform = xform.rotate(radians*180/Math.PI);
		// 	return rotate.call(context,radians);
		// };
		var translate = context.translate;
		context.translate = function(dx,dy){
			xform = xform.translate(dx,dy);
			return translate.call(context,dx,dy);
		};
		var transform = context.transform;
		context.transform = function(a,b,c,d,e,f){
			var m2 = svg.createSVGMatrix();
			m2.a=a; m2.b=b; m2.c=c; m2.d=d; m2.e=e; m2.f=f;
			xform = xform.multiply(m2);
			return transform.call(context,a,b,c,d,e,f);
		};
		var setTransform = context.setTransform;
		context.setTransform = function(a,b,c,d,e,f){
			xform.a = a;
			xform.b = b;
			xform.c = c;
			xform.d = d;
			xform.e = e;
			xform.f = f;
			return setTransform.call(context,a,b,c,d,e,f);
		};
		var pt  = svg.createSVGPoint();
		context.transformedPoint = function(x,y){
			pt.x=x; pt.y=y;
			return pt.matrixTransform(xform.inverse());
		}
	}