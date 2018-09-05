// Check for the various File API support.
if (window.File && window.FileReader && window.FileList && window.Blob) {
	// Great success! All the File APIs are supported.
} else {
	alert('The File APIs are not fully supported in this browser.');
}

// Check for the media API support.
if (!!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
	// Good to go!
} else {
	alert('getUserMedia() is not supported by your browser');
}

var speech, tab, iter = -1, textFile = null, logTime = new Array(), fileName, timeSave, startTime, endTime;
var recordedChunks = [];
var mediaRecorder;
var video = document.querySelector("#videoElement");

const constraints = {
  video: true,audio: true
};


function handleSuccess(stream) {
	video.srcObject = stream;
	var options = {audioBitsPerSecond : 128000, videoBitsPerSecond : 2500000 ,mimeType: "video/webm;codecs=vp8,opus"};
	mediaRecorder = new MediaRecorder(stream, options);
	mediaRecorder.ondataavailable = handleDataAvailable;
}

function handleError(error) {
	console.error('Rejected!', error);
}

function handleDataAvailable(event) {
	if (event.data.size > 0) {
		recordedChunks.push(event.data);
	} else {
		alert("not recording");
	}
}

function handleKeyboardEvent(event){
	if (event.defaultPrevented) {
    		return; // Should do nothing if the key event was already consumed.
  	}

  	switch (event.key) {
		case "Enter":
			if(iter > -1){
				logTime[iter] = timeSave +"@GOOD@" + iter + "@image" + iter + "\n";
				if(iter < 43){
					iter++;
					document.getElementById('toRead').src="../books/image" + iter + ".PNG";
				}
			}
			timeSave = getCurrentTime();
			break;
		case "ArrowRight":
			if(iter > -1){
				logTime[iter] = timeSave +"@BAD@" + iter + "@image" + iter + "\n";
				if(iter < 43){
					iter++;
					document.getElementById('toRead').src="../books/image" + iter + ".PNG";
				}
			}
			timeSave = getCurrentTime();	
   			break;
   		case "ArrowLeft":
			if(iter > 0){
				iter--;
   				document.getElementById('toRead').src="../books/image" + iter + ".PNG";
			}
			timeSave = getCurrentTime();	
   			break;
   		default:
   			return; // Quit when this doesn't handle the key event.
	}

  	// Consume the event for suppressing "double action".
  	event.preventDefault();
}

function handleBeginEvent(event){
	timeSave = getCurrentTime();
	startTime = getCurrentTime();
	iter = 1;
	//document.getElementById('toRead').innerHTML = "<strong>" + tab[iter] + "</strong>";
	document.getElementById('toRead').src="../books/image" + iter + ".PNG";
	fileName = "logTime_" + getCurrentTime();
	recordedChunks = []
	mediaRecorder.start(1000);
}

function handleEndEvent(event){
	endTime = getCurrentTime();
	makeTextFile(logTime,fileName);
	mediaRecorder.stop();
	
}

function download(event) {
	var blob = new Blob(recordedChunks, {type: 'video/webm'});
	var url = window.URL.createObjectURL(blob);
	var a = document.createElement('a');
  
	a.style = 'display: none';
	a.href = url;
	a.download = "Video_" + startTime + "TO" + endTime + ".webm";
	a.click();
	document.body.appendChild(a);
	window.URL.revokeObjectURL(url);
}

function makeTextFile(text,name) {
	var textToWrite = "";
	for(var i = 0 ; i < iter ; i++){
		textToWrite += logTime[i];
	}

	var data = new Blob([textToWrite], {type: 'text/plain'});

	// If we are replacing a previously generated file we need to
	// manually revoke the object URL to avoid memory leaks.
   	if (textFile !== null) {
		window.URL.revokeObjectURL(textFile);
	}

	textFile = window.URL.createObjectURL(data);

	// returns a URL you can use as a href
	var link = document.createElement('a');
	link.setAttribute('download', name);
	link.href = textFile;
	document.body.appendChild(link);

	// wait for the link to be added to the document
	window.requestAnimationFrame(function () {
		var event = new MouseEvent('click');
		link.dispatchEvent(event);
		document.body.removeChild(link);
	});
}

function getCurrentTime(){
	var a = new Date(Date.now());
	var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
	var year = a.getFullYear();
	var month = months[a.getMonth()];
	var date = a.getDate();
	var hour = a.getHours();
	var min = a.getMinutes();
	var sec = a.getSeconds();
	var millis = a.getMilliseconds();
	var time = date + '|' + month + '|' + year + '|' + hour + ':' + min + ':' + sec + ':' + millis;
	return time;
}

navigator.mediaDevices.getUserMedia(constraints).then(handleSuccess).catch(handleError);
window.addEventListener("keydown", handleKeyboardEvent, true);
document.getElementById('begin').addEventListener('click', handleBeginEvent, false);
document.getElementById('end').addEventListener('click', handleEndEvent, false);
document.getElementById('getVideo').addEventListener('click', download, false);

