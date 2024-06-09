# Make dataset
* Ground Truth
    * Parameters
        * U: 位置
        * R: 半径
        * Cx: 中心のx座標
        * Cy: 中心のy座標
* prior
    * 1,2,3,4: number_of_prior
    1. U,R,Cx,Cyをサンプリングする
        * どんな分布からサンプリングする？
            * 正規分布
    2. U,R,Cx,Cyを使ってデータを生成する
        * ここの点はどういう関数で定義される？
* observation
