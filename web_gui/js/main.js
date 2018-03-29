"use strict";

/*
 * Global variables.
 */
var container, stats;
var camera, scene, renderer;
var socket;

// The setup code.
window.onload = function() {
    init_3D();
    // init_Widgets();
    init_socket();
    needs_redisplay();
    };

function compute_orthographic_left_right_top_bottom( width, height )
{
    return [ -width/2, width/2, height/2, -height/2 ];
}
function init_3D() {

    container = document.createElement( 'div' );
    document.body.appendChild( container );

    var lrtb = compute_orthographic_left_right_top_bottom( window.innerWidth, window.innerHeight );
    camera = new THREE.OrthographicCamera( lrtb[0], lrtb[1], lrtb[2], lrtb[3], 1, 10000 );
    camera.position.z = 2;

    scene = new THREE.Scene();
    
    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setClearColor( 0xffffff );
    renderer.setSize( window.innerWidth, window.innerHeight );
    
    container.appendChild( renderer.domElement );

    stats = new Stats();
    stats.domElement.style.position = 'absolute';
    stats.domElement.style.top = '0px';
    stats.domElement.style.right = '0px';
    container.appendChild( stats.domElement );

    renderer.domElement.addEventListener( 'mousemove', onDocumentMouseMove, false );
    renderer.domElement.addEventListener( 'mousedown', onDocumentMouseDown, false );
    renderer.domElement.addEventListener( 'mouseup', onDocumentMouseUp, false );

    //

    window.addEventListener( 'resize', onWindowResize, false );
}

function onWindowResize() {
    needs_redisplay();
    
    // TODO: Update the camera.
    var lrtb = compute_orthographic_left_right_top_bottom( window.innerWidth, window.innerHeight );
    camera.left = lrtb[0];
    camera.right = lrtb[1];
    camera.top = lrtb[2];
    camera.bottom = lrtb[3];
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );
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
    requestAnimationFrame( function() {
        render();
        stats.update();
        } );
}

function render() {
    renderer.render( scene, camera );
}

function init_Widgets()
{
    // TODO: Fill this in for the GUI.
    
    $("#texture_map_URL").change( function() {
        load_texture( $(this).val() );
        // console.log( $(this).val() );
        } );
    $("#obj_URL").change( function() {
        load_OBJ( $(this).val() );
        // console.log( $(this).val() );
        } );
    
    $('input[name=mouse_mode]').change( function() {
        /// Recompute weights when we enter manipulate mode.
        if( $('input[name=mouse_mode]:checked').val() === 'mouse_manipulate_handles' )
        {
            compute_bbw_weights();
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
}


async function init_socket()
{
    socket = new WebSocketClient;
    await socket.connect('ws://localhost:9876');
    console.log( "Connected: ", socket.connected );
    
    // Call receive, which is an asynchronous function.
    receive();
}

async function receive()
{
    let data = await socket.receive();
    
    // TODO: Process the data. You can await more data if needed.
    
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
