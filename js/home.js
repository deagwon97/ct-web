const menu = document.querySelector('.navbar__menu');
const toggleBtn = document.querySelector('.navbar__toggleBtn');
const icons = document.querySelector('.navbar__icons');
const original_image = document.querySelector('#original_image');
const predict_image = document.querySelector('#predict_image');
const calculate = document.querySelector('.calculate_button');

toggleBtn.addEventListener('click', () => {
	menu.classList.toggle('active');
	icons.classList.toggle('active');
});

calculate.addEventListener('click', ()=>{
	original_image.classList.add('active');
	predict_image.classList.add('active');
});

let ct__image = document.getElementById('#upload_image').files[0];
let formData = new FormData();
formData.append("ct__image", ct__image);
fetch('/uploads/image', {method: "POST", body: formData});