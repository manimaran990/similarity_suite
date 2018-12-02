$(document).ready(function(e) {   
	$('.defect-container').click(function() {	 
	     $('.testcase-container').not($(this).next()).hide(200);
	     $(this).next('.testcase-container').toggle(400);
	});   
});