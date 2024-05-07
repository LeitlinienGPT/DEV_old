let fs = require('fs')

function read(path, encoding = null, flag = 'r') {
  return new Promise((resolve, reject) => {
    fs.readFile(path, { encoding, flag }, (err, data) => {
      if (!err) {
        resolve(data)
      } else {
        reject(err)
      }
    })
  })
}

function write(path, content, flag = 'w') {
  return new Promise((resolve, reject) => {
    fs.writeFile(path, content, { flag: flag, encoding: 'utf8' }, (err) => {
      if (err) {
        reject(err)
      } else {
        resolve()
      }
    })
  })
}

function mkdir(path) {
  return new Promise((resolve, reject) => {
    fs.mkdir(path, function(err) {
      if (err) {
        reject(err)
      } else {
        resolve('成功创建目录')
      }
    })
  })
}

function rename(oldPath, newPath) {
  return new Promise((resolve, reject) => {
    fs.rename(oldPath, newPath, (err, ) => {
      if (err) {
        reject(err)
      } else {
        resolve('修改成功')
      }
    })
  })
}

function readdir(path, options) {
  return new Promise((resolve, reject) => {
    fs.readdir(path, options, (err, files) => {
      if (err) {
        reject(err)
      } else {
        resolve(files)
      }
    })
  })
}

module.exports = {
  read,
  write,
  mkdir,
  rename,
  readdir,
}