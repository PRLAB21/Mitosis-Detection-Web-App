var fr; // variable to store the file reader
var is_img_ready = false;
var TAG = '[index.js]';
var SCREEN_WIDTH = 0;
var SCREEN_HEIGHT = 0;

$(function () {
    console.log(TAG, '[starts]');

    SCREEN_WIDTH = window.innerWidth;
    SCREEN_HEIGHT = window.innerHeight;

    $('#input-image-button').on('click', function (e) {
        $('#input-image').click();
    })

    $('#input-image').on('change', function (e) {
        e.preventDefault();
        console.log(e.target.files[0]);
        loadImage(e.target);
    })

    console.log(TAG, '[ends]');
});

// function to load the image from local path to img and canvas
function loadImage(img_src) {
    if (!img_src.files[0]) {
        alert('Please select an Image first!')
        return;
    }
    fr = new FileReader();
    fr.onload = updateImage;
    fr.readAsDataURL(img_src.files[0]);
}

function updateImage() {
    img = new Image();
    img.onload = function () {
        var canvas = document.getElementById('canvas');
        // console.log('[img.width]', img.width);
        // console.log('[img.height]', img.height);
        canvas.width = img.width;
        canvas.height = img.height;
        // console.log('[canvas.width]', canvas.width);
        // console.log('[canvas.height]', canvas.height);
        var context = canvas.getContext('2d');
        context.drawImage(img, 0, 0);
    };
    img.src = fr.result;
    is_img_ready = true;
}

function processImage() {
    let TAG = '[processImage]';
    console.log(TAG, '[starts]');
    if (!is_img_ready) {
        alert('No image to process!');
        return;
    }
    // send the image to the server and wait for a response
    canvas = document.getElementById('canvas');
    image_data = canvas.toDataURL('image/jpeg');
    classifier_name = $('#classifier-name').val();
    if (classifier_name === '--SELECT CLASSIFIER--') {
        alert('Select one classifier to perform inference!');
        return;
    }
    $.ajax({
        url: 'http://localhost:5000/inference',
        method: 'POST',
        contentType: 'application/json',
        crossDomain: true,
        data: JSON.stringify({
            image_data: image_data,
            classifier_name: classifier_name,
        }),
        beforeSend: function (data) {
            console.log(TAG, '[ajax][beforeSend]');
            $("#predictions").html('');
            $("#predictions").hide();
            $("#loading-gif").show();
        },
        success: function (data) {
            console.log(TAG + '[ajax][success]');
            // console.log(data);
            let score = Math.round(data['score'] * 100);
            let class_label = data['class_label'];
            // console.log('[score]', score);
            // console.log('[class_label]', class_label);
            $('#predictions').html(`${class_label} detected with ${score}% confidence`)
            $("#predictions").show();
            $('#loading-gif').hide();
        },
        error: function (err) {
            console.log(err)
        }
    });
}
