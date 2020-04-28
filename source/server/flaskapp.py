from flask import Flask, render_template, request
import json
import os
import personal_shopper
# import

app = Flask(__name__)

with open('./entity_thumnail.json', "r", encoding="utf8") as f:
    entities = json.load(f, encoding = 'utf-8')

with open('./seoulstore_tagged_05.json', "r", encoding="utf8") as f:
    item_data = json.load(f, encoding = 'utf-8')

identities = [x['name'] for x in entities]
thumb_filename_dict = {x: personal_shopper.get_thumb_name(x) for x in identities}

def goURL(msg, url):
    html = """
<script>
    alert("@msg");
    window.location.href = "@url";
</script>
        """
    html = html.replace("@msg", msg)
    html = html.replace("@url", url)
    return html

@app.route('/')
def index():
    return render_template('home.html', title='PERSONAL SHOPPER', entities=entities, length = len(entities))

@app.route('/view')      #/view?id=januaryrose_insta
def view():
    id = request.args.get("id")
    item_info = personal_shopper.get_items(id, item_data)
    personal_shopper.sort_by_distance(item_info, id)
    thumb_name = thumb_filename_dict[id]
    return render_template('view.html', id=id, thumb_name=thumb_name, item_info=item_info, length=len(item_info))

@app.route('/item_detail')    #/item_detail?id=000
def item_detail():
    id = request.args.get("id")
    item_info = personal_shopper.get_item_info(item_data, id)
    return render_template('item_detail.html', item_info=item_info, thumb_dict=thumb_filename_dict, length=len(item_info['likely']))


@app.route('/fileUpload', methods = ['POST'])
def fileUpload():
    f = request.files["file1"]
    title = request.form.get("title")
    algorithm = request.form.get("algorithm")
    algorithm = int(algorithm)
    src = './static/' + f.filename
    f.save(src)
    if algorithm == 0:
        processed = yolo3(src)
    else:
        processed = detectFace(src)
    path =  f.filename.split('.')[0] + '_processed' + '.jpg'
    cv.imwrite('./static/' + path, processed)
    id = listData[-1]['id'] + 1
    listData.append({"id": id, "img": path, "title": title})
    return goURL("업로드가 성공했습니다.", "/image")

@app.route('/deleteimage')  #/delete?id=0
def delete():
    del_id = request.args.get("id")
    for data in listData:
        if data['id'] == int(del_id): listData.remove(data)
    return goURL("자료를 삭제했습니다.", "/image")


if __name__ == '__main__':
    app.run(host='192.168.219.151', port=8484, debug=True)
    # app.run(host='0.0.0.0', port=8484, debug=True)
