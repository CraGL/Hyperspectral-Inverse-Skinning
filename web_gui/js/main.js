"use strict";

/*
 * Global variables.
 */
var container, stats;
var camera, scene, axes, renderer, controls;
var socket;
var draw_mode = "lines";

// The setup code.
window.onload = function() {
    init_3D();
    init_Widgets();
    init_socket();
    needs_redisplay();
    };

function init_3D() {

    container = document.createElement( 'div' );
    document.body.appendChild( container );

    /// Setup the camera here.
    init_camera();
    init_controls( camera );
    
    scene = new THREE.Scene();
    
    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setClearColor( 0xffffff );
    renderer.setSize( window.innerWidth, window.innerHeight );
    
    container.appendChild( renderer.domElement );

    stats = new Stats();
    stats.domElement.style.position = 'absolute';
    stats.domElement.style.top = '0px';
    stats.domElement.style.left = null;
    stats.domElement.style.right = '0px';
    container.appendChild( stats.domElement );

    renderer.domElement.addEventListener( 'mousemove', onDocumentMouseMove, false );
    renderer.domElement.addEventListener( 'mousedown', onDocumentMouseDown, false );
    renderer.domElement.addEventListener( 'mouseup', onDocumentMouseUp, false );

    window.addEventListener( 'resize', onWindowResize, false );
    
    // Add your initialization code here.
    axes = buildAxes( 1. );
    scene.add( axes );
}
function createAxisLine( position0, position1, color0, color1, dashed ) {
    // From: http://soledadpenades.com/articles/three-js-tutorials/drawing-the-coordinate-axes/
    var geom = new THREE.Geometry();
    var mat;
    
    if( dashed ) {
        mat = new THREE.LineDashedMaterial({ linewidth: 1, color: 0xffffff, vertexColors: THREE.VertexColors, dashSize: 10/255., gapSize: 5/255. });
    } else {
        mat = new THREE.LineBasicMaterial({ linewidth: 2, color: 0xffffff, vertexColors: THREE.VertexColors });
    }
    
    // Lines blend colors.
    // From: http://threejs.org/examples/webgl_lines_colors.html
    geom.colors = [ color0.clone(), color1.clone() ];
    
    geom.vertices.push( position0.clone() );
    geom.vertices.push( position1.clone() );
    // geom.computeLineDistances(); // This one is SUPER important, otherwise dashed lines will appear as simple plain lines
    
    var axis = new THREE.Line( geom, mat, THREE.LineSegments );
    axis.computeLineDistances = true;
    
    return axis;
}
function buildAxes( length, cube = true )
{
    // From: http://soledadpenades.com/articles/three-js-tutorials/drawing-the-coordinate-axes/
    var axes = new THREE.Object3D();
    axes.add( createAxisLine( new THREE.Vector3( 0, 0, 0 ), new THREE.Vector3( length, 0, 0 ), new THREE.Color( 0xFF0000 ), new THREE.Color( 0xFF0000 ), false ) ); // +X
    // axes.add( buildAxis( new THREE.Vector3( 0, 0, 0 ), new THREE.Vector3( -length, 0, 0 ), 0xFF0000, true) ); // -X
    axes.add( createAxisLine( new THREE.Vector3( 0, 0, 0 ), new THREE.Vector3( 0, length, 0 ), new THREE.Color( 0x00FF00 ), new THREE.Color( 0x00FF00 ), false ) ); // +Y
    // axes.add( buildAxis( new THREE.Vector3( 0, 0, 0 ), new THREE.Vector3( 0, -length, 0 ), 0x00FF00, true ) ); // -Y
    axes.add( createAxisLine( new THREE.Vector3( 0, 0, 0 ), new THREE.Vector3( 0, 0, length ), new THREE.Color( 0x0000FF ), new THREE.Color( 0x0000FF ), false ) ); // +Z
    // axes.add( buildAxis( new THREE.Vector3( 0, 0, 0 ), new THREE.Vector3( 0, 0, -length ), 0x0000FF, true ) ); // -Z
    
    if( cube ) {
        // XY
        axes.add( createAxisLine( new THREE.Vector3( length, 0, 0 ), new THREE.Vector3( length, length, 0 ), new THREE.Color( 0xAAAAAA ), new THREE.Color( 0xAAAAAA ), false ) );
        axes.add( createAxisLine( new THREE.Vector3( 0, length, 0 ), new THREE.Vector3( length, length, 0 ), new THREE.Color( 0xAAAAAA ), new THREE.Color( 0xAAAAAA ), false ) );
        
        // YZ
        axes.add( createAxisLine( new THREE.Vector3( 0, length, 0 ), new THREE.Vector3( 0, length, length ), new THREE.Color( 0xAAAAAA ), new THREE.Color( 0xAAAAAA ), false ) );
        axes.add( createAxisLine( new THREE.Vector3( 0, 0, length ), new THREE.Vector3( 0, length, length ), new THREE.Color( 0xAAAAAA ), new THREE.Color( 0xAAAAAA ), false ) );
        
        // XZ
        axes.add( createAxisLine( new THREE.Vector3( length, 0, 0 ), new THREE.Vector3( length, 0, length ), new THREE.Color( 0xAAAAAA ), new THREE.Color( 0xAAAAAA ), false ) );
        axes.add( createAxisLine( new THREE.Vector3( 0, 0, length ), new THREE.Vector3( length, 0, length ), new THREE.Color( 0xAAAAAA ), new THREE.Color( 0xAAAAAA ), false ) );
        
        // XY from 1,1,1
        axes.add( createAxisLine( new THREE.Vector3( length, length, length ), new THREE.Vector3( length, length, 0 ), new THREE.Color( 0xAAAAAA ), new THREE.Color( 0xAAAAAA ), false ) );
        
        // YZ from 1,1,1
        axes.add( createAxisLine( new THREE.Vector3( length, length, length ), new THREE.Vector3( 0, length, length ), new THREE.Color( 0xAAAAAA ), new THREE.Color( 0xAAAAAA ), false ) );
        
        // XZ from 1,1,1
        axes.add( createAxisLine( new THREE.Vector3( length, length, length ), new THREE.Vector3( length, 0, length ), new THREE.Color( 0xAAAAAA ), new THREE.Color( 0xAAAAAA ), false ) );
    }
    
    return axes;
}
function init_camera() {
    camera = new THREE.PerspectiveCamera( 45, window.innerWidth / window.innerHeight, 1, 10 );
    // camera = new THREE.OrthographicCamera( window.innerWidth / - 2, window.innerWidth / 2, window.innerHeight / 2, window.innerHeight / - 2, 1, 10 );
    camera.position.set( .5, .5, 3 );
    camera.lookAt( new THREE.Vector3( .5, .5, .5 ) );
    // camera = new THREE.OrthographicCamera( -1, 2, 2, -1, 1, 4 );
    // camera.position.z = 4;
}
function init_controls( camera ) {
    var origin = new THREE.Vector3( .5, .5, .5 );
    
    controls = new THREE.TrackballControls( camera );
    controls.target.copy( origin );
    
    controls.rotateSpeed = 1.0;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.8;

    controls.noZoom = true;
    controls.noPan = true;
    // Keep the horizon level? No.
    camera.noRoll = false;

    controls.staticMoving = true;
    // dynamicDampingFactor only has an effect when staticMoving is false.
    // Setting it to 0 allows constant-speed rotation in a loop.
    controls.dynamicDampingFactor = 0.0;

    // We need this because we wouldn't otherwise call render().
    // Namely, we do not call render() in our animate() function called
    // many times per second.
    controls.addEventListener( 'change', render );
    
    // Set up the inertial rotation checkbox.
    // document.getElementById( 'intertia' ).addEventListener( 'change', function() { controls.staticMoving = !document.getElementById('intertia').checked; needs_redisplay(); } );
    // document.getElementById( 'intertia' ).checked = !controls.staticMoving;
    
    document.getElementById( 'look_from_111' ).addEventListener( 'click', function() {
        // First reset the controls (keep the current origin).
        var origin = controls.target.clone();
        controls.reset();
        controls.target.copy( origin );
        
        var targetToCameraLength = camera.position.clone().sub( controls.target ).length();
        
        camera.position.set( 1, 1, 1 );
        camera.position.sub( controls.target ).setLength( targetToCameraLength ).add( controls.target );
        
        needs_redisplay();
        } );
}

function onWindowResize() {
    needs_redisplay();
    
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    
    renderer.setSize( window.innerWidth, window.innerHeight );
    
    controls.handleResize();
}

function onDocumentMouseMove( event ) {
    needs_redisplay();
    
    event.preventDefault();
}

function onDocumentMouseDown( event ) {
    needs_redisplay();
    
    event.preventDefault();
}

function onDocumentMouseUp( event ) {
    needs_redisplay();
    
    event.preventDefault();
}

function needs_redisplay() {
    requestAnimationFrame( animate );
}

function animate() {
    if( !controls.staticMoving ) requestAnimationFrame( animate );
    
    controls.update();
    // We have to split these because controls may call render() directly.
    render();
}

function render() {
    renderer.render( scene, camera );
    stats.update();
}

function init_Widgets()
{
    // TODO: Fill this in for the GUI.
    
    /*
    
    $("#texture_map_URL").change( function() {
        // load_texture( $(this).val() );
        // console.log( $(this).val() );
        } );
    $("#obj_URL").change( function() {
        // load_OBJ( $(this).val() );
        // console.log( $(this).val() );
        } );
    
    $('input[name=mouse_mode]').change( function() {
        /// Recompute weights when we enter manipulate mode.
        if( $('input[name=mouse_mode]:checked').val() === 'mouse_manipulate_handles' )
        {
            // compute_bbw_weights();
            needs_redisplay();
        }
        } );
    
    $('#visualize_weights').change( function() {
        needs_redisplay();
        
        if( the_mesh === undefined ) return;
        
        var val = $('#visualize_weights :selected').val();
        if( val === 'off' )
        {
            the_mesh_container.remove( the_mesh );
            the_mesh = new THREE.Mesh( the_mesh.geometry.clone(), create_default_material() );
            the_mesh_container.add( the_mesh );
            return;
        }
        
        var handle_index = parseInt( val );
        // If we're still in add mode, we may not yet have weights.
        if( the_weights === null || handle_index >= the_weights[0].length )
        {
            console.error( "Can't visualize the weights because they haven't been computed yet." );
            return;
        }
        
        // If we're here, then we should be good.
        the_mesh_container.remove( the_mesh );
        the_mesh = new THREE.Mesh( the_mesh.geometry.clone(), create_weights_material( handle_index ) );
        the_mesh_container.add( the_mesh );
        } );
    
    $('#clear_handles').click( function() {
        reset_handles();
        } );
    $('#reset_transforms').click( function() {
        reset_transforms();
        } );
    
    $('#save_weights').click( function() {
        var globals = save_weights();
        var blob = new Blob([ JSON.stringify( globals ) ], {type: "application/json"});
        saveAs( blob, "skinning_weights.json" );
        } );
    $('#load_weights').change( function( evt ) {
        var files = evt.target.files;
        if( files.length === 0 )
        {
            console.error( "Load weights: No file loaded." );
            return;
        }
        else if( files.length > 1 )
        {
            console.error( "Load weights can't load more than one file at a time." );
            return;
        }
        
        var file = files[0];
        // UPDATE: For some reason, the file.type is never application/json on Windows.
        if( false )
        // if( file.type !== 'application/json' )
        {
            console.error( "Load weights can only accept JSON files." );
            return;
        }
        
        var reader = new FileReader();
        reader.onload = function( loaded_event ) {
            var globals = JSON.parse( loaded_event.target.result );
            load_weights( globals );
            };
        reader.readAsText( file );
        } );
    
    // Set up small-link handle presets.
    $(".small-link.handle_preset").on( 'click', function( evt ) {
        // Setup the handles.
        var changed = setup_handles( JSON.parse( $(this).attr('data-handle-preset') ) );
        if( changed )
        {
            /// Recompute the weights and switch to manipulate mode.
            compute_bbw_weights();
            $( "#mouse_manipulate_handles" ).prop( "checked", true );
        }
        } );
    
    // Set up small-link URL's to "change" their respective "for" tags.
    $(".small-link.URL").on( 'click', function( evt ) { $( '#' + $(this).attr('for') ).val( $(this).attr('href') ).change(); } );
    
    // By default, use the grid texture.
    $("#texture_map_URL_grid").click();
    // By default, use the circle OBJ.
    $("#obj_URL_circle").click();
    */
}


async function init_socket()
{
    socket = new WebSocketClient;
    
    // Socket from URL in modern JavaScript: https://stackoverflow.com/questions/979975/how-to-get-the-value-from-the-get-parameters
    var url = new URL( window.location.href );
    var port = 9876;
    if( url.searchParams.has("port") ) {
        port = parseInt( url.searchParams.get("port") );
    }
    
    await socket.connect('ws://localhost:' + port);
    console.log( "Connected on port " + port + ": ", socket.connected );
    
    // Call receive, which is an asynchronous function.
    receive();
}

function createLineFromReceived( position, direction, color0, width = 2, len = 2 ) {
    // From: http://soledadpenades.com/articles/three-js-tutorials/drawing-the-coordinate-axes/
    var geom = new THREE.Geometry();
    var mat = new THREE.LineBasicMaterial({ linewidth: width, color: 0xffffff, vertexColors: THREE.VertexColors });
        
    direction = direction.normalize().multiplyScalar( len );
    geom.vertices.push( position.clone().sub( direction ) );
    geom.vertices.push( position.clone().add( direction ) );
    geom.colors = [ color0.clone(), color0.clone() ];
    
    var line = new THREE.Line( geom, mat, THREE.LineSegments );
    line.computeLineDistances = true;
        
    return line;
}

function createPointFromReceived( pts, color0, size = 2 ) {
	// From: https://github.com/josdirksen/learning-threejs/blob/master/chapter-07/10-create-particle-system-from-model.html
    var geom = new THREE.Geometry();
    
	for (var i = 0; i < pts.length; i++) { 
		var point = new THREE.Vector3( pts[i][0], pts[i][1], pts[i][2] );
		geom.vertices.push( point );
	}
	
	var material = new THREE.PointsMaterial({
		color: color0,
		size: 0.05,
		transparent: false,
		blending: THREE.AdditiveBlending,
		// map: generateSprite()
	});
	var cloud = new THREE.Points(geom, material);
	
	cloud.sortParticles = true;
	return cloud;
}

var all_pts = null;
async function receive()
{
    let data = await socket.receive();
    
    // TODO: Process the data. You can await more data if needed.
    {
        // console.log( data );
        if( data == "\"lines\"" ) {
        	draw_mode = "lines";
        	console.log( "switch to lines." );
        }
        else if( data == "\"points\"" ) {
        	draw_mode = "points";
        	console.log( "switch to points." );
        	scene.remove( axes );
        	axes = buildAxes( 1., false );
        	scene.add( axes );
            controls.target.set( 0,0,0 );
        }
        
        data = await socket.receive();
        
		// Create the data holder.
		if( all_pts !== null ) scene.remove( all_pts );
		all_pts = new THREE.Object3D();

		// Assume data is an array of 3D positions.
		let pts = JSON.parse( data );

		if( draw_mode == "lines" ) {
			for (var i = 0; i < pts.length; i += 2) { 
				// let pt = pts[i];
				var position = new THREE.Vector3( pts[i][0], pts[i][1], pts[i][2] );
				var direction = new THREE.Vector3( pts[i + 1][0], pts[i + 1][1], pts[i + 1][2] );
				if ( i >= pts.length - 2 ) all_pts.add( createLineFromReceived( position, direction, new THREE.Color( 0xc441f4 ), 5, 100 ) );
				else 					   all_pts.add( createLineFromReceived( position, direction, new THREE.Color( 0x000000 ) ) );
			}
		}
		else if( draw_mode == "points" ) {
			var points = createPointFromReceived( pts, new THREE.Color( 0xc441f4 ), 2 );
			all_pts.add( points );
		}
    }
    // Add the points to the scene.
    scene.add( all_pts );
    needs_redisplay();
    
    // Call ourselves recursively.
    receive();
}

// From: http://stackoverflow.com/questions/15313418/javascript-assert
function assert(condition, message) {
    if (!condition) {
        message = message || "Assertion failed";
        if (typeof Error !== "undefined") {
            throw new Error(message);
        }
        throw message; // Fallback
    }
}
