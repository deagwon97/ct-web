let {PythonShell} = require('python-shell');
var package_name = 'pytube'
let options = {
    args : ['pydicom', 'opencv-python', 'pydicom', 'torch', 'segmentation-models-pytorch']
}
PythonShell.run('./install_package.py', options, 
    function(err, results)
    {
        if (err) throw err;
        else console.log(results);
    })