let PythonShell = require('python-shell');
let options = {
  mode: 'text',
  pythonPath: '../python/',
  pythonOptions: ['-u'],
  scriptPath:'',
  args:['-img_path',
        'uploads/image.dcm',
        '-model_path',
        '../python/model.pth']
}

PythonShell.run('inference.py', options, function (err, results){
  if (err) throw err;
  console.log('results: %j', results);
});