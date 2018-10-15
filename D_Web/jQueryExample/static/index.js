window.onload = function() {
    alert( "welcome" );
};

$( document ).ready(function() {
    $( "a"
    ).addClass( "test"
    //    niubi.

    ).click(function( event ) {
        // alert( "Thanks for visiting!" );
        alert( "The link will no longer take you to jquery.com" );
        event.preventDefault();
    //    niubi.
        $( this ).hide( "slow" );
    })


});