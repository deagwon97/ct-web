const express = require("express");
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { resolve } = require('path');
const { response } = require('express');
const router = express.Router();
//var inference = require("./inference.js");

try {
  fs.readdirSync('uploads');
} catch (error) {
  console.error('uploads 폴더가 없어 uploads 폴더를 생성합니다.');
  fs.mkdirSync('uploads');
}

const upload = multer({
  storage: multer.diskStorage({
    destination(req, file, cb){
      cb(null, 'uploads/')
    },
    filename(req, file, done) {
      const ext = path.extname(file.originalname);
      done(null, "input"+ext);
    },

  }),
  limits : {fileSize: 10 * 1024 * 1024},
});


let {PythonShell} = require('python-shell');
let options = {
  mode: 'text',
  pythonPath: "C:\\Users\\user\\AppData\\Local\\Continuum\\anaconda3\\envs\\deep\\python.exe",
  pythonOptions: ['-u'],
  scriptPath:'../python',
  args:['-img_path',
        '../js/uploads/input.dcm',
        '-model_path',
        '../python/model.pth',
        '-save_path',
        '../js/predictions/']
}




router.post('/uploads/image', upload.single('ct__image'), (req, res) => {
  //다운로드가 끝났습니다.  
  PythonShell.run('inference.py', options, function (err, results){
    if (err) throw err;
      console.log('results: %j', results);
    });
  res.status(204).send();
});



module.exports = router;